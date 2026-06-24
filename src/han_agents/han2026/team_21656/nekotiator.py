import random 
import json 
import re 
import warnings 
from attrs import define ,field 
from typing import Any 
from negmas_llm import OllamaNegotiator 
from negmas .sao import ResponseType ,SAOResponse ,SAOState 
from negmas .common import Outcome ,Value 
from negmas .preferences .base_ufun import BaseUtilityFunction 
from negmas .gb import GBState 
from negmas .gb .components .genius .models import GFSEGABayesianModel 





class Nekotiator (OllamaNegotiator ):
    def __init__ (self ,temperature :float =0.3 ,timeout :float =60.0 ,num_retries :int =1 ,max_tokens :int =512 ,**kwargs ):
        llm_kwargs =kwargs .pop ("llm_kwargs",{})
        llm_kwargs .setdefault ("timeout",timeout )
        llm_kwargs .setdefault ("num_retries",num_retries )
        super ().__init__ (
        temperature =temperature ,
        max_tokens =max_tokens ,
        llm_kwargs =llm_kwargs ,
        **kwargs 
        )

        self ._my_min =0.0 
        self ._my_max =1.0 
        self ._best_ever =None 
        self ._last_my_offer =None 
        self ._n_issues :int =field (init =False ,default =0 )



        self .opponent_weights ={}
        self ._target_weight_sum =1.0 

        self ._llm_weights :dict [int ,float ]=field (factory =dict )

        self ._opponent_has_spoken =False 
        self ._is_uniform_scenario =False 


        self ._opponent_peaks ={}
        self ._last_seen_offer =None 

        self ._best_opp_offer_for_me =None 
        self ._best_opp_offer_util =self ._my_min 
        self ._n_step_rate =0 

        self ._opp_first_util_for_me =None 
        self .issue_type ={}
        self ._is_zero_sum =False 
        self ._zero_sum_issues ={}
        self .mymodel =MyModel ()
        self .mymodel .set_negotiator (self )


    def on_preferences_changed (self ,changes ):
        state =self .nmi .state if self .nmi else None 
        if state is None or state .step ==0 :
            if self .ufun and self .nmi :
                self .candidates =list (self .nmi .outcome_space .enumerate_or_sample (1000 ))

                _mn =self .ufun .min ()
                self ._my_min =float (_mn )if _mn is not None else 0.0 

                _mx =self .ufun .max ()
                self ._my_max =float (_mx )if _mx is not None else 1.0 

                self ._best_ever =self .ufun .best ()
                self ._last_my_offer =None 

                self ._llm_weights ={}
                self ._n_issues =len (self .nmi .outcome_space .issues )
                self ._llm_has_real_evidence =False 


                res =self .ufun .reserved_value 
                if res is not None :
                    self ._my_min =max (self ._my_min ,float (res ))


                self ._opponent_peaks ={}
                self ._last_seen_offer =None 

                self ._best_opp_offer_for_me =None 
                self ._best_opp_offer_util =self ._my_min 

                self ._opp_first_util_for_me =None 

                self ._is_zero_sum =self ._is_linear_zero_sum ()
                self ._zero_sum_issues ={issue .name :self ._issue_is_linear_zero_sum (issue )for issue in self .nmi .outcome_space .issues }

                self ._llm_preferences ={}
                for issue in self .nmi .outcome_space .issues :
                    is_zero_sum =self ._zero_sum_issues .get (issue .name ,False )
                    values =list (issue .values )
                    default_target ="Any"if is_zero_sum else (values [0 ]if values else "Any")
                    self ._llm_preferences [issue .name ]={
                    "target":default_target ,
                    "importance":2 ,
                    "weight":5 
                    }

                self .candidates =list (self .nmi .outcome_space .enumerate_or_sample (1000 ))



    def __call__ (self ,state :SAOState ,dest :str |None =None )->SAOResponse :
        self ._n_step_rate =state .step /self .nmi .n_steps 


        chat_text ="How about these conditions? Please let me know if you have any specific requests."

        if state .current_offer :
            current_offer =state .current_offer 
            current_util_for_me =float (self .ufun (current_offer ))

            self .mymodel .update (state ,current_offer ,state .current_proposer )


            opponent_msg =state .current_data .get ("text","")if (state .current_data and isinstance (state .current_data ,dict ))else ""


            if opponent_msg :
                context_prompt =self .build_system_prompt (state ,opponent_msg ,self ._is_zero_sum )
                response_text =self ._send_to_llm (opponent_msg ,require_json =True ,state =state )
                llm_prefs ,llm_weights ,chat_text =self ._parse_llm_response (response_text )


                if llm_prefs :

                    chat_text +=" I have created a new proposal taking your requests into consideration."
                    for issue_name ,pref_data in llm_prefs .items ():
                        self ._llm_preferences [issue_name ]=pref_data 

                if llm_weights :
                    self ._llm_weights =self ._normalize_llm_weights (llm_weights )
                elif hasattr (self ,'_llm_preferences')and self ._llm_preferences :
                    self ._llm_weights =self ._derive_llm_weights_from_preferences ()
            else :

                chat_text ="Thank you for your proposal. I have made some adjustments and prepared a counter-offer."


            if current_util_for_me >self ._best_opp_offer_util :
                self ._best_opp_offer_util =current_util_for_me 
                self ._best_opp_offer_for_me =current_offer 

            if self ._opp_first_util_for_me is None :
                self ._opp_first_util_for_me =current_util_for_me 

            aspiration =self ._get_concession_aspiration (state ,current_util_for_me )
            my_next_proposal =self ._generate_win_win_proposal (aspiration )
            my_next_util =float (self .ufun (my_next_proposal ))


            if opponent_msg and hasattr (self ,'_llm_preferences')and self ._llm_preferences :
                if not getattr (self ,'_is_accommodated',True ):
                    chat_text ="I have carefully considered your request, but unfortunately it is difficult to meet due to our internal constraints. I have prepared an alternative proposal instead. How does this look?"


            if current_util_for_me >=aspiration or current_util_for_me >=my_next_util :
                accept_text ="This is an excellent proposal! It meets our aspiration threshold, so I am happy to accept it."
                return SAOResponse (ResponseType .ACCEPT_OFFER ,current_offer ,data ={"text":accept_text })


            if state .relative_time >0.95 and self ._best_opp_offer_for_me :
                last_resort_text ="Time is running out, and I really want to avoid walking away without an agreement. Could we settle on this previous condition you offered?"
                return SAOResponse (ResponseType .REJECT_OFFER ,self ._best_opp_offer_for_me ,data ={"text":last_resort_text })


            return SAOResponse (ResponseType .REJECT_OFFER ,my_next_proposal ,data ={"text":chat_text })

        else :

            first_turn_text ="Thank you for the opportunity to negotiate. I will start by proposing my preferred conditions. Please let me know if there are any specific terms you prioritize."
            return SAOResponse (ResponseType .REJECT_OFFER ,self ._best_ever ,data ={"text":first_turn_text })



    def _get_concession_aspiration (self ,state :SAOState ,current_opp_offer_util :float |None =None )->float :
        """Concede from 0.7 down to 0.5 over time while keeping a floor."""
        if state .step ==0 :
            return min (self ._my_max ,1.0 )

        START_ASPIRATION =min (0.7 ,self ._my_max )
        FLOOR_ASPIRATION =max (0.5 ,self ._my_min )
        progress =min (1.0 ,state .relative_time **2 )

        if current_opp_offer_util is not None and self ._opp_first_util_for_me is not None :

            concession_gain =max (0.0 ,current_opp_offer_util -self ._opp_first_util_for_me )
            concession_factor =min (1.0 ,concession_gain /max (1e-6 ,self ._my_max -self ._my_min ))
            progress =min (1.0 ,progress +concession_factor *0.3 )

        aspiration =START_ASPIRATION -(START_ASPIRATION -FLOOR_ASPIRATION )*progress 
        return max (FLOOR_ASPIRATION ,min (START_ASPIRATION ,aspiration ))

    def build_system_prompt (self ,state :SAOState ,opponent_msg :str ,is_zero_sum :bool )->str :
        my_offer_str =str (self ._last_my_offer )if self ._last_my_offer else "None"
        opp_offer_str =str (self ._last_seen_offer )if self ._last_seen_offer else "None"


        opponent_model_str ="None"
        if hasattr (self ,'_llm_preferences')and self ._llm_preferences :
            model_lines =[]
            for i ,issue in enumerate (self .nmi .outcome_space .issues ):
                pref =self ._llm_preferences .get (issue .name )
                if pref :
                    weight =self ._llm_weights .get (i ,0.0 )if self ._llm_weights else 0.0 
                    target =pref .get ('target','Any')
                    importance =pref .get ('importance',2 )
                    model_lines .append (f"{issue.name}: weight={weight:.1f}, target={target}, importance={importance}")
            if model_lines :
                opponent_model_str ="\n    ".join (model_lines )


        env_rule =(
        "ZERO-SUM: Opponent gain is your loss. Use weights 1-10."
        if is_zero_sum else 
        "NON-ZERO-SUM: Seek trade-offs. Use weights 1-10 and targets."
        )

        return f"""You are a negotiation assistant.
Context: {"Zero-Sum" if is_zero_sum else "Non-Zero-Sum"}
{env_rule}
Your last offer: {my_offer_str}
Opponent last offer: {opp_offer_str}
Opponent message: "{opponent_msg}"

Previous opponent model:
{opponent_model_str}

Task:
- Estimate opponent preferences for all issues.
- Return JSON with weight 1-10, importance 1-3, target or Any.
- Non-zero-sum: target values matter.
- Zero-sum: higher values are better.

Return EXACT JSON only:
{{
  "opponent_model": {{
    "IssueName": {{ "weight": 1-10, "target": "TargetValue or Any", "importance": 1-3 }}
  }},
  "predicted_best_offer": "OutcomeStringOrNone",
  "reasoning": "Short explanation",
  "text": "Reply text"
}}

Outcome space:
{self.format_outcome_space(state)}
"""

    def _parse_llm_response (self ,response_text :str ):
        extracted_prefs ={}
        opponent_weights ={}
        chat_text ="Let's find a win-win agreement."
        predicted_best_offer =None 

        try :
            json_match =re .search (r"\{[\s\S]*\}",response_text )
            if json_match :
                data =json .loads (json_match .group ())


                print (f"\n=== LLM RAW RESPONSE DEBUG ===")
                print (f"LLM full JSON: {json.dumps(data, indent=2)}")


                opponent_model =data .get ("opponent_model",{})
                if opponent_model :
                    print (f"Opponent model extracted: {opponent_model}")

                    for issue_name ,model_data in opponent_model .items ():
                        extracted_prefs [issue_name ]={
                        "target":model_data .get ("target","Any"),
                        "importance":model_data .get ("importance",2 ),
                        "weight":model_data .get ("weight",5 )
                        }
                        opponent_weights [issue_name ]=model_data .get ("weight",5 )


                extracted_prefs_old =data .get ("extracted_preferences",{})
                opponent_weights_old =data .get ("opponent_weights",{})
                if extracted_prefs_old :
                    extracted_prefs .update (extracted_prefs_old )
                if opponent_weights_old :
                    opponent_weights .update (opponent_weights_old )

                print (f"Extracted preferences: {extracted_prefs}")
                print (f"Opponent weights (raw): {opponent_weights}")

                chat_text =data .get ("text",chat_text )
                predicted_best_offer =data .get ("predicted_best_offer",None )
        except Exception as e :
            warnings .warn (f"JSON decomposition failed: {str(e)}")

        return extracted_prefs ,opponent_weights ,chat_text 

    def _send_to_llm (self ,opponent_msg :str ,**kwargs )->str :
        state =kwargs .get ("state")
        is_zero_sum =getattr (self ,'_is_zero_sum',False )
        sys_prompt =self .build_system_prompt (state ,opponent_msg ,is_zero_sum )

        messages =[
        {"role":"system","content":sys_prompt },
        {"role":"user","content":"Analyze the context and generate the JSON response."}
        ]
        return self ._call_llm (messages ,require_json =kwargs .get ("require_json",False ))

    def _normalize_llm_weights (self ,weights :dict )->dict [int ,float ]:
        """1-10スケールのLLM重みを正規化（合計1.0）"""
        normalized ={}
        raw ={}


        print (f"\n=== NORMALIZE LLM WEIGHTS DEBUG ===")
        print (f"Raw input weights: {weights}")

        for issue_name ,val in weights .items ():
            if isinstance (val ,str ):
                v =val .strip ().lower ()

                if v in ("high","10","10.0"):
                    raw [issue_name ]=10.0 
                elif v in ("very_high","8","8.0"):
                    raw [issue_name ]=8.0 
                elif v in ("medium_high","6","6.0"):
                    raw [issue_name ]=6.0 
                elif v in ("medium","med","5","5.0"):
                    raw [issue_name ]=5.0 
                elif v in ("medium_low","3","3.0"):
                    raw [issue_name ]=3.0 
                elif v in ("low","1","1.0"):
                    raw [issue_name ]=1.0 
                else :
                    try :
                        w =float (v )
                        raw [issue_name ]=max (1.0 ,min (10.0 ,w ))
                    except Exception :
                        raw [issue_name ]=5.0 
            else :
                try :
                    w =float (val )
                    raw [issue_name ]=max (1.0 ,min (10.0 ,w ))
                except Exception :
                    raw [issue_name ]=5.0 

        print (f"Raw converted: {raw}")
        total =sum (raw .values ())
        print (f"Total raw sum: {total}")

        if total <=0 :
            return {i :1.0 /max (1 ,self ._n_issues )for i in range (self ._n_issues )}


        for i ,issue in enumerate (self .nmi .outcome_space .issues ):
            if issue .name in raw :
                normalized [i ]=raw [issue .name ]/total 
            else :

                for raw_name in raw :
                    if raw_name in issue .name or issue .name in raw_name :
                        print (f"  Fuzzy match: '{raw_name}' -> issue '{issue.name}'")
                        normalized [i ]=raw [raw_name ]/total 
                        break 

        print (f"Normalized: {normalized}")

        if not normalized :
            return {i :1.0 /max (1 ,self ._n_issues )for i in range (self ._n_issues )}

        return normalized 

    def _derive_llm_weights_from_preferences (self )->dict [int ,float ]:
        raw ={}
        for i ,issue in enumerate (self .nmi .outcome_space .issues ):
            pref =self ._llm_preferences .get (issue .name ,{})
            try :
                importance =float (pref .get ('importance',2 ))
            except Exception :
                importance =2.0 
            raw [issue .name ]=min (max (importance ,1.0 ),3.0 )
        total =sum (raw .values ())
        if total <=0 :
            return {i :1.0 /max (1 ,self ._n_issues )for i in range (self ._n_issues )}
        return {i :raw [issue .name ]/total for i ,issue in enumerate (self .nmi .outcome_space .issues )}

    def _issue_value_ratio (self ,issue ,value )->float :
        try :
            values =list (issue .values )
            if not values :
                return 0.0 
            if value not in values :
                try :
                    values =[float (v )for v in values ]
                except Exception :
                    pass 
            if value in values :
                idx =values .index (value )
                return idx /max (1 ,len (values )-1 )
            try :
                normalized =float (value )
                numeric_values =[float (v )for v in values ]
                mn =min (numeric_values )
                mx =max (numeric_values )
                if mx -mn <=0 :
                    return 0.0 
                return (normalized -mn )/(mx -mn )
            except Exception :
                return 0.0 
        except Exception :
            return 0.0 

    def _target_value_score (self ,issue ,value ,target )->float :
        """Return a peaked score centered on target, decreasing towards the edges."""
        value_norm =self ._issue_value_ratio (issue ,value )
        target_norm =self ._issue_value_ratio (issue ,target )
        distance =abs (value_norm -target_norm )
        return max (0.0 ,1.0 -distance *distance )

    def _generate_win_win_proposal (self ,aspiration :float )->Outcome :
        """LLMの相手モデルを使用してスコアリングし、最適な提案を生成"""

        valid_for_me =[c for c in self .candidates if float (self .ufun (c ))>=aspiration ]


        print (f"\n=== PROPOSAL GENERATION DEBUG ===")
        print (f"Aspiration: {aspiration}")
        print (f"Total candidates: {len(self.candidates)}")
        print (f"Valid for me (util >= {aspiration}): {len(valid_for_me)}")


        print (f"\n=== LLM OPPONENT MODEL ===")
        if hasattr (self ,'_llm_preferences')and self ._llm_preferences :
            print (f"LLM Preferences: {self._llm_preferences}")
            print (f"LLM Weights: {self._llm_weights}")
            print (f"Zero-sum issues: {self._zero_sum_issues}")
        else :
            print (f"WARNING: No _llm_preferences found!")

        if not valid_for_me :
            print (f"WARNING: No candidates meet aspiration. Returning best_ever.")
            return self ._best_ever 


        def opponent_utility (offer :Outcome )->float :
            """相手の推定効用を計算"""
            total_util =0.0 
            if not (hasattr (self ,'_llm_preferences')and self ._llm_preferences ):
                print (f"  DEBUG opponent_utility: No _llm_preferences available")
                return 0.5 


            is_debug_offer =offer ==valid_for_me [0 ]if valid_for_me else False 
            if is_debug_offer :
                print (f"\n  DEBUG: Detailed calculation for first offer {offer}:")

            for i ,issue in enumerate (self .nmi .outcome_space .issues ):
                pref =self ._llm_preferences .get (issue .name )
                if not pref :
                    if is_debug_offer :
                        print (f"    Issue {i} ({issue.name}): No preference found, skipping")
                    continue 

                issue_weight =self ._llm_weights .get (i ,0.1 )
                target =pref .get ('target','Any')
                is_zero_sum_issue =self ._zero_sum_issues .get (issue .name ,False )

                value_at_offer =offer [i ]
                normalized_value =self ._issue_value_ratio (issue ,value_at_offer )

                if is_debug_offer :
                    print (f"    Issue {i} ({issue.name}):")
                    print (f"      Value: {value_at_offer}, Normalized: {normalized_value:.3f}")
                    print (f"      Weight: {issue_weight:.3f}, Target: {target}, Zero-sum: {is_zero_sum_issue}")

                if is_zero_sum_issue :

                    contrib =issue_weight *normalized_value 
                    total_util +=contrib 
                    if is_debug_offer :
                        print (f"      Zero-sum contrib: {issue_weight:.3f} * {normalized_value:.3f} = {contrib:.3f}")
                else :

                    if target !='Any':
                        score =self ._target_value_score (issue ,value_at_offer ,target )
                        contrib =issue_weight *score 
                        total_util +=contrib 
                        if is_debug_offer :
                            print (f"      Non-zero-sum (target={target}) score: {score:.3f}, contrib: {contrib:.3f}")
                    else :

                        contrib =issue_weight *normalized_value 
                        total_util +=contrib 
                        if is_debug_offer :
                            print (f"      Non-zero-sum (no target) contrib: {issue_weight:.3f} * {normalized_value:.3f} = {contrib:.3f}")

            if is_debug_offer :
                print (f"    Total opponent utility: {total_util:.3f}")

            return total_util 



        def opponent_score (offer :Outcome )->float :
            return opponent_utility (offer )


        valid_sorted =sorted (valid_for_me ,key =opponent_score ,reverse =True )


        print (f"\nTop 3 candidates (by opponent utility):")
        for idx ,offer in enumerate (valid_sorted [:3 ]):
            my_util =float (self .ufun (offer ))
            opp_util =opponent_score (offer )
            print (f"  {idx+1}. {offer} | my_util={my_util:.3f}, opp_util={opp_util:.3f}")

        my_next =valid_sorted [0 ]

        print (f"Selected offer: {my_next}")
        print (f"  My utility: {float(self.ufun(my_next)):.3f}")
        print (f"  Opponent utility: {opponent_score(my_next):.3f}")

        self ._last_my_offer =my_next 
        return my_next 


    def _update_llm_model (self ,llm_prefs :dict ):
        """LLMの重要度評価(1-3)を統計モデルの重みに反映する"""
        if not llm_prefs :
            return 


        raw_weights ={}
        for issue_name ,pref in llm_prefs .items ():

            importance =float (pref .get ("importance",2 ))
            raw_weights [issue_name ]=importance **2 

        total =sum (raw_weights .values ())
        if total >0 :

            for i ,issue in enumerate (self .nmi .outcome_space .issues ):
                if issue .name in raw_weights :

                    self .mymodel ._issue_weights [i ]=raw_weights [issue .name ]/total 


            total_weight =sum (self .mymodel ._issue_weights .values ())
            for i in self .mymodel ._issue_weights :
                self .mymodel ._issue_weights [i ]/=total_weight 
    def _issue_is_linear_zero_sum (self ,issue )->bool :
        values =list (issue .values )
        if len (values )<2 :
            return False 


        try :

            issue_index =next (i for i ,iss in enumerate (self .nmi .outcome_space .issues )if iss is issue or iss .name ==issue .name )
        except StopIteration :
            issue_index =None 

        print (f"\n=== ZERO-SUM CHECK for {getattr(issue,'name',None)} ===")
        print (f"  values: {values}")

        if issue_index is not None :


            try :

                eval_values =values 
                try :
                    if len (values )==2 :
                        nv =[float (v )for v in values ]
                        if all (float (v ).is_integer ()for v in nv ):
                            a =int (nv [0 ])
                            b =int (nv [1 ])
                            if b -a >=2 and b -a <=100 :
                                eval_values =list (range (a ,b +1 ))
                                print (f"  expanded endpoint values -> {eval_values}")
                except Exception as e :
                    print (f"  expansion failed: {e}")
                    eval_values =values 

                utils =[float (self .get_value_utility (issue_index ,v ))for v in eval_values ]
                print (f"  eval_values: {eval_values}")
                print (f"  utils: {utils}")

                if len (eval_values )<=2 :

                    print (f"  only {len(eval_values)} points available; cannot infer linear zero-sum from endpoints")
                    print (f"  -> Falling back to label-based check")

                    raise ValueError ("Not enough points for ufun-based check")

                mn =min (utils )
                mx =max (utils )
                print (f"  mn={mn}, mx={mx}, diff={mx - mn}")

                if mx -mn >1e-8 :
                    diffs =[utils [i +1 ]-utils [i ]for i in range (len (utils )-1 )]
                    avg_diff =sum (diffs )/len (diffs )
                    print (f"  diffs: {diffs}, avg_diff={avg_diff}")
                    res =all (abs (d -avg_diff )<1e-5 for d in diffs )
                    print (f"  result from ufun-based check: {res}")
                    return res 
                else :

                    print ("  ufun returned identical utilities; treating as non-zero-sum")
                    return False 
            except Exception as e :

                print (f"  ufun-based check failed: {e}")


        print (f"  -> Using fallback (label-based check)")
        try :
            numeric_values =[float (v )for v in values ]
            print (f"  numeric_values: {numeric_values}")
            diffs =[numeric_values [i +1 ]-numeric_values [i ]for i in range (len (numeric_values )-1 )]
            avg_diff =sum (diffs )/len (diffs )
            print (f"  diffs: {diffs}, avg_diff={avg_diff}")
            res =all (abs (d -avg_diff )<1e-5 for d in diffs )
            print (f"  result from label-based check: {res}")
            return res 
        except Exception as e :

            print (f"  label-based check failed: {e}, len(values)={len(values)}")
            res =len (values )==2 
            print (f"  fallback result (len==2): {res}")
            return res 

    def _is_linear_zero_sum (self )->bool :
        """論点の値が数値的で、等間隔に並んでいるか（線形/ゼロサム的か）を判定する"""
        if not hasattr (self ,'nmi')or not self .nmi or not hasattr (self .nmi ,'outcome_space'):
            return False 

        zero_sum =True 
        for issue in self .nmi .outcome_space .issues :
            issue_ok =self ._issue_is_linear_zero_sum (issue )
            self .issue_type [issue .name ]=issue_ok 
            if not issue_ok :
                zero_sum =False 
        return zero_sum 

    def _predict_opponent_best_offer (self )->Outcome |None :
        """LLMの相手モデルを使用して、相手の最良提案を予測"""
        if not hasattr (self ,'_llm_preferences')or not self ._llm_preferences :
            return None 

        def opponent_utility (offer :Outcome )->float :
            """相手の推定効用を計算"""
            total_util =0.0 
            for i ,issue in enumerate (self .nmi .outcome_space .issues ):
                pref =self ._llm_preferences .get (issue .name )
                if not pref :
                    continue 

                issue_weight =self ._llm_weights .get (i ,0.1 )
                target =pref .get ('target','Any')
                is_zero_sum_issue =self ._zero_sum_issues .get (issue .name ,False )

                normalized_value =self ._issue_value_ratio (issue ,offer [i ])

                if is_zero_sum_issue :

                    total_util +=issue_weight *normalized_value 
                else :

                    if target !='Any':
                        score =self ._target_value_score (issue ,offer [i ],target )
                        total_util +=issue_weight *score 
                    else :
                        total_util +=issue_weight *normalized_value 

            return total_util 


        best_opp_candidate =max (self .candidates ,key =opponent_utility ,default =None )
        return best_opp_candidate 


    def get_value_utility (self ,issue_index :int ,value ,base_offer :Outcome |None =None )->float :
        """指定した論点(issue_index)で値(value)にしたときの全体効用を返す。
        base_offer が与えられなければ `self._best_ever` をベースにする。
        Outcome をリスト化して該当論点だけ差し替えて `ufun` を評価する。
        安全のため例外時は 0.0 を返す。
        """
        if base_offer is None :
            base_offer =self ._best_ever if getattr (self ,'_best_ever',None )is not None else (self .candidates [0 ]if getattr (self ,'candidates',None )else None )
        if base_offer is None :
            return 0.0 

        try :
            offer =list (base_offer )
            offer [issue_index ]=value 
            return float (self .ufun (offer ))
        except Exception :
            return 0.0 

    def get_normalized_value_utility (self ,issue_index :int ,value ,base_offer :Outcome |None =None )->float :
        """同一論点内の値を 0..1 に正規化した効用を返す。
        候補値の最小/最大で正規化。等しい場合は 0.5 を返す。
        """
        vals =list (self .nmi .outcome_space .issues [issue_index ].values )
        if not vals :
            return 0.5 

        utils =[self .get_value_utility (issue_index ,v ,base_offer )for v in vals ]
        mn =min (utils )
        mx =max (utils )
        if mx <=mn :
            return 0.5 
        u =self .get_value_utility (issue_index ,value ,base_offer )
        return (u -mn )/(mx -mn )





    @property 
    def my_ufun (self ):
        return self .mymodel 


class MyModel (GFSEGABayesianModel ):
    learning_coef :float =0.2 
    learning_value_addition :int =1 

    def _initialize (self ):
        super ()._initialize ()
        self ._last_opponent_bid =None 

    def update (self ,state :GBState ,offer :Outcome ,partner_id :str )->None :
        """統計的な更新を行った後、LLMの情報があれば重みを上書きする"""
        super ().update (state ,offer ,partner_id )



        negotiator =self .negotiator 

        if hasattr (negotiator ,'_llm_preferences')and negotiator ._llm_preferences :
            self ._apply_llm_weights (negotiator ._llm_preferences )

    def _apply_llm_weights (self ,llm_prefs :dict ):
        """LLMからの重要度(importance)を重みに適用する"""

        raw_weights ={}
        for issue_name ,pref in llm_prefs .items ():
            importance =float (pref .get ("importance",2 ))
            raw_weights [issue_name ]=(importance /3.0 )

        total =sum (raw_weights .values ())
        if total >0 :

            for i ,issue in enumerate (self .negotiator .nmi .outcome_space .issues ):
                if issue .name in raw_weights :

                    self ._issue_weights [i ]=raw_weights [issue .name ]/total 


            total_weight =sum (self ._issue_weights .values ())
            for i in self ._issue_weights :
                self ._issue_weights [i ]/=total_weight 