# Diffusion-PID-Protection
Implementation of the paper "PID: Prompt-Independent Data Protection Against Latent Diffusion Models"

TODO:
- [x] Initialize the repo. (2024/5/28)
- [x] Training scripts & implementation of the PID.
- [x] Implementation of the evaluation code.
- [x] Visualizations & training data.
- [] Implementation of the baselines (FSGM, ASPL, AdvDM).

### Protecting images with PID
```sh
  sh PID.sh
```


### Fine-tuning
```sh
  sh train_dreambooth.sh # DreamBooth

  sh train_drembooth_lora.sh # LoRA
```

### Evaluation
```sh
  sh evaluate.sh
```

### Citation
```
Please consider citing our work if you find it helpful!
```

### Acknowledgment
This repo uses some of the code from the links below. We sincerely admire their great work!
- https://huggingface.co/docs/diffusers/en/training/dreambooth
- https://huggingface.co/docs/diffusers/en/training/lora
- https://github.com/VinAIResearch/Anti-DreamBooth/tree/main
- https://github.com/psyker-team/mist
- https://github.com/chaofengc/IQA-PyTorch
- https://github.com/timesler/facenet-pytorch
- https://github.com/krshrimali/No-Reference-Image-Quality-Assessment-using-BRISQUE-Model/tree/master/Python
