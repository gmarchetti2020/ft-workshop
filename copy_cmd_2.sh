gsutil cp -r  gs://shivajid-xporf/p* .
gsutil cp -r gs://shivaji-gemma/custom/lora/en-fr/* .
gsutil cp gs://cloud-samples-data/vertex-ai/model-evaluation/peft_train_sample.jsonl .
cp gemma_lora_trainer_safetensor_v3.py gemma_lora_trainer_safetensor_v4.py 
gsutil cp gs://shivaji-gemma/gemma3_1b_allenia_math_v2/* 
gsutil cp gs://shivaji-gemma/gemma3_1b_allenia_math_v3/* gs://the-fine-tuners/gemma3_1b_openmath/draft_release/
cp ../Dockerfile .
cp data.py trainer/gemma2b/
cp data.py trainer/gemma3b/
cp ../Dockerfile .
cp data.py trainer/gemma2b/
cp data.py trainer/gemma3b/
cp data.py ../ft-workshop/
cp gemma3_1b_it_fft_trainer.py ../ft-workshop/
cp test_prompts.txt ../ft-workshop/test_prompt_medqa.txt 
cp ../tensor_crunch/requirements.txt .
cp ../tensor_crunch/data.py /
cp ../tensor_crunch/gemma3_1b_it_fft_trainer.py .
cp ../tensor_crunch/requirements.txt requirements_tunix.txt
cp ../tensor_crunch/scripts/create_python_env.sh create_tunix_python_env.sh
cp ../tensor_crunch/docs/gemma3_1b_fft_trainer_guide.md .
hitory|grep cp >> copy_cmd_.sh
history|grep cp >> copy_cmd_.sh
awk '/cp|gsutil/ {sub(/.*(cp|gsutil)/, "\\1"); print}' copy_cmd_.sh 
hitory|grep cp >> copy_cmd_.sh
history|grep cp >> copy_cmd_.sh
