
# chatGPT

## Methods
We trained this model using Reinforcement Learning from Human Feedback (RLHF), using the same methods as InstructGPT, but with slight differences in the data collection setup. We trained an initial model using supervised fine-tuning: human AI trainers provided conversations in which they played both sides—the user and an AI assistant. We gave the trainers access to model-written suggestions to help them compose their responses. We mixed this new dialogue dataset with the InstructGPT dataset, which we transformed into a dialogue format.

To create a reward model for reinforcement learning, we needed to collect comparison data, which consisted of two or more model responses ranked by quality. To collect this data, we took conversations that AI trainers had with the chatbot. We randomly selected a model-written message, sampled several alternative completions, and had AI trainers rank them. Using these reward models, we can fine-tune the model using Proximal Policy Optimization. We performed several iterations of this process.

## step
- collect demonstration data and train a supervised policy
  - a prompt is sampled from our prompt dataset
  - a labeler demonstrates the desired output behavior
  - this data is used to fine-tune GPT-3.5 with supervised learning
```
    explain reinforcement learning to a 6 year old
                      |
                     \|/
      we give treats and punishments to teach...
                      |
                     \|/
                     SFT
```
- collect comparison data and traain a reward model
  - a prompt and several model outputs are sampled
  - a labeler ranks the outputs from best or worst
  - this data is used to train our reward model
- optimize a policy against the reward model using the PPO reinforcement learning algorithm
  - a new prompt is sampled from the dataset
  - the PPO model is initialized from the supervised policy
  - the policy generates an output
  - the reward model calculates a reward fro the output
  - the reward is used to tupdate the policy using PPO

## GPT-3.5
- ChatGPT is fine-tuned from a model in the GPT-3.5 series, which finished training in early 2022. You can learn more about the 3.5 series here. ChatGPT and GPT 3.5 were trained on an Azure AI supercomputing infrastructure.

## Limitations
- ChatGPT sometimes writes plausible-sounding but incorrect or nonsensical answers. Fixing this issue is challenging, as: (1) during RL training, there’s currently no source of truth; (2) training the model to be more cautious causes it to decline questions that it can answer correctly; and (3) supervised training misleads the model because the ideal answer depends on what the model knows, rather than what the human demonstrator knows.
- ChatGPT is sensitive to tweaks to the input phrasing or attempting the same prompt multiple times. For example, given one phrasing of a question, the model can claim to not know the answer, but given a slight rephrase, can answer correctly.
- The model is often excessively verbose and overuses certain phrases, such as restating that it’s a language model trained by OpenAI. These issues arise from biases in the training data (trainers prefer longer answers that look more comprehensive) and well-known over-optimization issues.12
- Ideally, the model would ask clarifying questions when the user provided an ambiguous query. Instead, our current models usually guess what the user intended.
- While we’ve made efforts to make the model refuse inappropriate requests, it will sometimes respond to harmful instructions or exhibit biased behavior. We’re using the Moderation API to warn or block certain types of unsafe content, but we expect it to have some false negatives and positives for now. We’re eager to collect user feedback to aid our ongoing work to improve this system.

## Iterative deployment
- Today’s research release of ChatGPT is the latest step in OpenAI’s iterative deployment of increasingly safe and useful AI systems. Many lessons from deployment of earlier models like GPT-3 and Codex have informed the safety mitigations in place for this release, including substantial reductions in harmful and untruthful outputs achieved by the use of reinforcement learning from human feedback (RLHF).

## reference
- https://www.paperdigest.org/2023/01/recent-papers-on-chatgpt/
- https://proceedings.neurips.cc/paper/2020
- Learning to summarize from human feedback
  - https://arxiv.org/pdf/2009.01325.pdf
  - https://github.com/openai/summarize-from-feedback
- Scaling Laws for Reward Model Overoptimization
  - https://arxiv.org/pdf/2210.10760.pdf
- https://ajl.org/bugs.
- Toward Trustworthy AI Development: Mechanisms for Supporting Verifiable Claims
  - https://arxiv.org/pdf/2004.07213.pdf
- Bias Bounty Programs as a Method of Combatting Bias in AI
  - https://rubinovitz.com/2018/08/01/bias-bounty-programs-as-a-method-of-combatting/
