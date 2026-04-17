"""core.cognitive_os — Cognitive OS namespace (redirect stub).

The cognitive engine previously lived at ``core.cognitive_os.engine``
inside this repository.  It has been extracted to a standalone repo
with a permissive license (MIT-0) so it can be used in contexts
requiring OSI-approved licensing (e.g. the ARC Prize 2026 /
ARC-AGI-3 competition, which mandates CC0- or MIT-0-equivalent terms).

    Engine repo:  https://github.com/khub-ai/cognitive-os-engine
    Install:      pip install git+https://github.com/khub-ai/cognitive-os-engine
    Import as:    import cognitive_os

KF itself remains under PolyForm Noncommercial 1.0.0.  Future KF code
(robotics adapters, etc.) that wants to build on the engine should
install ``cognitive-os-engine`` as a dependency and import the
``cognitive_os`` package directly — the previous
``core.cognitive_os.engine`` path is no longer available.
"""

raise ImportError(
    "core.cognitive_os.engine has been extracted to a standalone "
    "repository under MIT-0 license.  Install "
    "'pip install git+https://github.com/khub-ai/cognitive-os-engine' "
    "and import the top-level 'cognitive_os' package instead."
)
