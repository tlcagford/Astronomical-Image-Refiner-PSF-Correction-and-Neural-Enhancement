# AI / parameter advisor stubs.
# Replace `ai_suggest` with calls to your LLM or a rules engine.

def ai_suggest_params(metadata, goal="science-grade"):
    """
    Return a dict of suggested params given metadata.
    This is a rules-first placeholder — replace with real LLM call.
    """
    inst = metadata.get("INSTRUME", "").lower()
    if "nircam" in inst:
        return {"pixscale": 0.03, "drizzle": "auto", "cr_reject": "science"}
    if "miri" in inst:
        return {"pixscale": 0.11, "drizzle": "native", "cr_reject": "standard"}
    if "wfc3" in inst:
        return {"psf_sampling": "native", "choose_stars": "ai-auto", "output": "psf_model.fits"}
    # default
    return {"pixscale": 0.05}
