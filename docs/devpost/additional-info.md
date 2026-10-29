# Devpost: additional questions

- **Track:** Physical AI
- **Did you use Tavily?** Yes. It grounds every object's size and mass (advanced search, `/extract` for full pages, cross-source agreement), material friction and bounciness, and the Franka gripper's limits; these numbers decide what is simulated and which tasks are refused. See the README section "How we use Tavily".
- **Submitter type:** Individual
- **Country / province:** Canada / British Columbia
- **Demo URL:** https://austintransformer-spec-to-sim.hf.space (free, no sign-up; recorded runs replay without any model calls, live runs are capped per visitor)
- **Repo:** https://github.com/atabak-eghbal/spec-to-sim (MIT)
- **Video:** https://youtu.be/HfrEUVrve4M

## AI tools used (be explicit; judges value honesty here)
I built Spec-to-Sim with Claude Code (Anthropic) as a coding partner: it wrote and refactored much of the code, ran the evaluations and the simulations, built the demo video pipeline, and ran simulated reviewer passes whose findings I acted on. I set the direction, made the design decisions (what to build, what to cut, e.g. dropping Ultra after the data), tested the app myself and reviewed the results. Inside the product, the AI is NVIDIA Nemotron on Nebius Token Factory (Super, Nano, Super VL) and Tavily for web grounding.

(Adjust this paragraph to your own words before submitting.)

## Learning level / career value
Fill in from your own experience. Suggested angle: first end-to-end Physical AI system combining an LLM cascade, web grounding and physics simulation with a rigorous evaluation; directly relevant to robotics simulation and evaluation tooling roles.
