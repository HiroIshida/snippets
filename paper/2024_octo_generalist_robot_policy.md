## octo model (points)
- Arbitrary input token (created from observations and tasks) to output token (decoded into action)
- With no additional training, accept different camera configuration, different joint types,
- Accept NLP instructions, image goal
- 800k demonstrations of Open X-embodiement

## pretraining design choice
- less inductive bias
- wide variety of policy objectives

## Good
- retraining is easy
    - do not require retraining the transformer
    - only trains positional encoding / action head / lightweight encoder

## Comparison to existing methods
- RT-X: user must stick to action / observation space used in the pretraining

## Method
- architecture: 
    - input tokenizer (language instructions / goals / observations) => tokens
    - task and observations toeknizer  (language /observations / goals) => tokens
        - use t5-base model to tokenize langauage
        - image is processed by a shallow CNN and then extract patches
    - lightweight action head implements diffusion policy
        - this action head predicts a chunk of actions. [98, 97]
- training:
    - use RT-X dataset
    - objective: predict continuous multi-modal action distributions [34, 17]
    - diffusion head objective: DDPM loss [34] (see 17 for details)

## Evaluation
- CMU baking:
    - 50% agains 25% of resnet + transformer scratch
    - pickup, open door, put down, close door
    - 120 demonstrations
    - 15hz
    - 
- Stanford coffee:
    - 75% agains 45% of resnet + transformer scratch
    - pickup one of four objects, then place it inside of a coffee machine
    - 118 demonstrations
    - 15hz
    
- Berkeley insertion:
    - 70% agains 10% of resnet + transformer scratch
    - insert peg into hole (matching tolerance is only 1.5mm)
    - 100 demonstrations
    - 5hz
    
- success rate:
    - finetuning しないと簡単なタスクでも50%程度
    - finetuning,
        - force-feedbackも入力にいれてfinetuning
        - jointも考慮.
- speed:
    - A500 GPU: 5 hours to finetuning
    - RTX4090: 10Hz inference

- Question:
    - 言語/画像以外のゴールを入力することはできるのか?
    - 時系列の表現方法?

## Note
    - diffusionではなくL2lossをするとうまくいかない.

## 松嶋さんのDL輪読会 
https://www.youtube.com/watch?v=oVGANsnAezo&ab_channel=%E6%9D%BE%E5%B0%BE%E7%A0%94%E7%A9%B6%E5%AE%A4DL%E8%BC%AA%E8%AA%AD%E4%BC%9A
