## Skills
A skill is a local instruction bundle stored in `SKILL.md`.

### Available skills
- popcorn-submission-workflow: Helps with Popcorn CLI registration, submission setup, submission modes, and file directives. (file: /Users/21sk/amd_202602/mxfp4-mm/.popcorn/skills/popcorn-submission-workflow/SKILL.md)
- load-inline-native-code: Helps write CUDA and HIP kernels using torch.utils.cpp_extension.load_inline(). Use when writing native GPU code inside a Python submission. (file: /Users/21sk/amd_202602/mxfp4-mm/.popcorn/skills/load-inline-native-code/SKILL.md)

### How to use skills
- Load the skill by reading its `SKILL.md` file when user requests match the description.
- Follow progressive disclosure: read only relevant referenced files/scripts as needed.
- Keep the workspace setup aligned with `popcorn setup`.
