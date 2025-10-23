# Pipeline initialization for SD3.5 GRPO training
from .sd3_pipeline_with_logprob import pipeline_with_logprob

def setup_sd3_pipeline_for_grpo(pipeline):
    """
    Attach the pipeline_with_logprob method to an SD3 pipeline instance.
    This enables log probability tracking during sampling.
    """
    import types
    pipeline.pipeline_with_logprob = types.MethodType(pipeline_with_logprob, pipeline)
    return pipeline
