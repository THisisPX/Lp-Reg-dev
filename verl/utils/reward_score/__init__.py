# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

import traceback

from . import prime_math


def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    zero=False,
    use_compute_score_v2=False,
):
    # if data_source == "openai/gsm8k":
    #     from . import gsm8k

    #     res = gsm8k.compute_score(solution_str, ground_truth)
    # elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
    #     from . import math

    #     res = math.compute_score(solution_str, ground_truth)
    #     # [Optional] Math-Verify Integration
    #     # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
    #     # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
    #     # To use it, override the `compute_score` function with the following implementation:

    #     # from . import math_verify
    #     # res = math_verify.compute_score(solution_str, ground_truth)
    # elif data_source == "math_dapo" or data_source.startswith("aime"):
    #     from . import math_dapo

    #     res = math_dapo.compute_score(solution_str, ground_truth)
    # elif data_source in [
    #     "numina_aops_forum",
    #     "numina_synthetic_math",
    #     "numina_amc_aime",
    #     "numina_synthetic_amc",
    #     "numina_cn_k12",
    #     "numina_olympiads",
    #     "math",
    # ]:
    #     from . import prime_math

    #     res = prime_math.compute_score(solution_str, ground_truth)
    # elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
    #     # Use the passed sandbox_fusion_url if available
    #     if sandbox_fusion_url:
    #         from . import sandbox_fusion

    #         # Pass the URL directly, ground_truth likely contains test cases here
    #         res = sandbox_fusion.compute_score(sandbox_fusion_url, concurrent_semaphore, solution_str, ground_truth, continuous=True)
    #     else:
    #         # If no sandbox URL is provided, fall back to prime_code or raise error
    #         from . import prime_code

    #         # Assuming prime_code doesn't need the URL
    #         res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    # elif data_source in ["hiyouga/geometry3k"]:
    #     from . import geo3k

    #     res = geo3k.compute_score(solution_str, ground_truth)
    # else:d
    try:
        if data_source in {"codecontests", "apps", "codeforces", "taco", "prime_code", "code"}:
            if sandbox_fusion_url:
                from . import sandbox_fusion

                res = sandbox_fusion.compute_score(
                    sandbox_fusion_url,
                    concurrent_semaphore,
                    solution_str,
                    ground_truth,
                    continuous=True,
                )
            else:
                from . import prime_code

                res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
        elif zero:
            if not use_compute_score_v2:
                res = prime_math.compute_score(solution_str, str(ground_truth))
            else:
                res = prime_math.compute_score_v2(solution_str, str(ground_truth))
        else:
            res = prime_math.compute_score_v2(solution_str, str(ground_truth)) if use_compute_score_v2 else prime_math.compute_score(solution_str, str(ground_truth))
            # print(f"data_source: {data_source}")
            # raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])
    except Exception as e:
        print(f"[ERROR] Error in process_completion for task : {str(e)}")
        traceback.print_exc()  # 打印完整堆栈
        raise  # 重新抛出异常以便上层捕获


def _summarize_code_metadata(metadata_list):
    if not metadata_list:
        return 1.0, 1.0
    status_values = []
    for metadata in metadata_list:
        if isinstance(metadata, dict):
            status_values.append(metadata.get("status"))
    if not status_values:
        return 1.0, 1.0
    compile_failure_status = {"compile_error", "compile_timeout", "compile_error_skipped"}
    compile_success = True
    test_success = True
    for status in status_values:
        if status in compile_failure_status:
            compile_success = False
        if status != "success":
            test_success = False
    return float(compile_success), float(test_success)


def _default_compute_score_with_extra_info(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    zero=False,
    use_compute_score_v2=False,
):
    try:
        is_code_task = data_source in {"codecontests", "apps", "codeforces", "taco", "prime_code", "code"}
        if is_code_task:
            if sandbox_fusion_url:
                from . import sandbox_fusion

                score, metadata_list = sandbox_fusion.compute_score(
                    sandbox_fusion_url,
                    concurrent_semaphore,
                    solution_str,
                    ground_truth,
                    continuous=True,
                )
            else:
                from . import prime_code

                score, metadata_list = prime_code.compute_score(solution_str, ground_truth, continuous=True)
            compile_success, test_success = _summarize_code_metadata(metadata_list)
            return {
                "score": float(score),
                "acc": test_success,
                "compile_success": compile_success,
                "test_success": test_success,
            }

        if zero:
            score = prime_math.compute_score_v2(solution_str, str(ground_truth)) if use_compute_score_v2 else prime_math.compute_score(solution_str, str(ground_truth))
        else:
            score = prime_math.compute_score_v2(solution_str, str(ground_truth)) if use_compute_score_v2 else prime_math.compute_score(solution_str, str(ground_truth))
        
        # For non-code tasks, we still return a dict to keep reward_extra_info list lengths consistent
        return {
            "score": float(score),
            "acc": 1.0,
            "compile_success": 1.0,
            "test_success": 1.0,
        }
    except Exception as e:
        print(f"[ERROR] Error in process_completion for task : {str(e)}")
        traceback.print_exc()
        raise
