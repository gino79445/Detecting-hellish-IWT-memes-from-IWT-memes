# Detecting hellish IWT memes from IWT memes

<div align=center>
<img src="https://github.com/gino79445/Detecting-hellish-IWT-memes-from-IWT-memes/blob/main/fig.png?raw=true" style="width:500px" />
</div>


Social network platforms have witnessed the increasing importance of visual communication. Images and multimedia content play a crucial role in engaging users, and among these, IWT memes have emerged as a popular form of communication. However, the prevalence of harmful or offensive content in these memes can have a negative impact on society. In this paper, we propose a novel multimodal approach for identifying hellish IWT memes. Our approach leverages features extracted from IWT memes, including visual, text, and facial features, to accurately identify these harmful memes. 

## EXAMPLE: Hellish IWT Meme and Normal IWT Meme
<div align=center>
<img src="https://github.com/gino79445/Detecting-hellish-IWT-memes-from-IWT-memes/blob/main/1.jpg?raw=true" style="width:150px" />
 &emsp;&emsp;
<img src="https://github.com/gino79445/Detecting-hellish-IWT-memes-from-IWT-memes/blob/main/2.jpg?raw=true" style="width:150px" />
</div>

## Data Preparation
[Download link](https://drive.google.com/drive/folders/1y_BlSEha4aTCeKKUO9pz8ngYIOomaPvk?usp=sharing)


***hell_npy_val*** : The data in "hell_npy_val" has been preprocessed for dataloader.

***training*** : images and labels.

## Result
<div align=center>
  
||Accuracy|Precision|Recall|F1-score|
|---:|:---:|---:|---:|---:|
|Vision|0.722|0.688|0.697|0.690|
|Vision + Text|0.731|0.697|0.711|0.703|
|Vision + Text + face|0.737|0.701|0.731|0.713|
  
</div>












