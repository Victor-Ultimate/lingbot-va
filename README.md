<h1 align="center">LingBot-VA: Causal World Modeling for Robot Control</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2601.21998"><img src="https://img.shields.io/static/v1?label=Paper&message=PDF&color=red&logo=arxiv"></a>
  <a href="https://technology.robbyant.com/lingbot-va"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="https://huggingface.co/collections/robbyant/lingbot-va"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=orange"></a>
  <a href="https://modelscope.cn/collections/Robbyant/LingBot-VA"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=purple"></a>
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/License-Apache--2.0-green"></a>
</p>

<p align="center">
  <img src="assets/teaser_v3.png" width="100%">
</p>



https://github.com/user-attachments/assets/cec7b7a6-953b-4fa4-8f1a-47efc1fce547




## 💫 Meet **LingBot-VA**!  We've built an AR diffusion framework for simultaneous world modeling and action! 🤖✨

**LingBot-VA** has focused on:
- **Autoregressive Video-Action World Modeling**: Architecturally unifies visual dynamics prediction and action inference within a single interleaved sequence while maintaining their conceptual distinction.
- **High-efficiency Execution**: A dual-stream mixture-of-transformers(MoT) architecture with Asynchronous Execution and KV Cache.
- **Long-Horizon Performance and Generalization**: High improvements in sample efficiency, long-horizon success rates, and generalization to novel scenes.

# 🚀 News
- **[2026-01-29]** Weights and code for shared backbone released! Please stay tuned for our separated version!




---



# 📦 Model Download
- **Pretrained Checkpoints for Post-Training**

| Model Name | Huggingface Repository | ModelScope Repository  | Description |
| :--- | :--- | :--- | :--- |
| lingbot-va-base &nbsp; | [🤗 robbyant/lingbot-va-base &nbsp;](https://huggingface.co/robbyant/lingbot-va-base) | [🤖 Robbyant/lingbot-va-base &nbsp;](https://modelscope.cn/models/Robbyant/lingbot-va-base)  | LingBot-VA w/ shared backbone|
| lingbot-va-posttrain-robotwin &nbsp; | [🤗 robbyant/lingbot-va-posttrain-robotwin &nbsp;](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin) | [🤖 Robbyant/lingbot-va-posttrain-robotwin &nbsp;](https://modelscope.cn/models/Robbyant/lingbot-va-posttrain-robotwin)  | LingBot-VA-Posttrain-Robotwin w/ shared backbone|
---

# 🛠️ Quick Start

## Installation
**Requirements**
 • Python == 3.10.16
 • Pytorch == 2.9.0
 • CUDA 12.6

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==5.0.0 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install flash-attn --no-build-isolation
```


## Deploying LingBot-VA for Inference
LingBot-VA supports both standalone execution and Server-Client architecture which separates the model environment from simulation. By isolating dependencies, the design avoids package clashes and supports distributed inference on GPUs, clusters, and other devices.

<!-- ### Standalone  Inference
```python
python inference.py
```
This processes the example data from `examples/0/` and saves visualizations to `result/`. -->

### Evaluation on RoboTwin-2.0

**Preparing the Environment**

You can follow the official instructions from the original RoboTwin-2.0 repository:  
[https://robotwin-platform.github.io/doc/usage/robotwin-install.html](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)

 **Deploying the Inference Server**
```bash
# single GPU
bash evaluation/robotwin/launch_server.sh

# multi-GPU
bash evaluation/robotwin/launch_server_multigpus.sh
```

 **Executing the Inference Client**
```bash
# single GPU
task_name="adjust_bottle";
save_root="results/";
bash evaluation/robotwin/launch_client.sh ${task_name} ${save_root}

# multi-GPU
save_root="results/"
task_group_id=0;
bash evaluation/robotwin/launch_client_multigpus.sh ${save_root} ${task_group_id}
```

Related experiments results will be save in `/path/to/your/RoboTwin/${save_root}`. Please note that an `eval_result` folder is also generated. This is a native output from RoboTwin and is identical to the contents in the results folder; it can be safely ignored.
It is important to note that the inference server and client must be deployed on the same machine. For launching multi-GPU client, we padded the original 50 tasks to 56 via duplication and partitioned them into 7 groups to align with the 8-GPU configuration of our inference node. You can specify the `task_group_id` (0-6) to select a particular group for inference. For detailed grouping configurations, please refer to `evaluation/robotwin/launch_client_multigpus.sh`.

### Run Image to Video-Action Generation

We also provide a script for image to video-action generation:

```bash
CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh 
```



---

# 📊 Performance

We evaluate our model on both simulation benchmarks and real-world scenarios, and achieve state-of-the-art performance.

## Simulation Evaluation

- **RoboTwin 2.0**

We are the first to propel RoboTwin 2.0 metrics performance past the 90+ threshold！
<table style="border-collapse: collapse; width: auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; font-size: 13px; line-height: 1.2;">
<!-- 指标说明 -->
  <p style="font-size: 12px; color: #666; margin-bottom: 5px;">* All metrics are reported in percentage (%). Higher values are <b>bolded</b>.</p>
  <thead>
    <tr style="border-top: 2px solid black; border-bottom: 1px solid black;">
      <th align="left" style="padding: 6px 12px; white-space: nowrap;">Method (Average 50 Tasks)</th>
      <th align="center" style="padding: 6px 12px;">Easy SR (%)</th>
      <th align="center" style="padding: 6px 12px;">Hard SR (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 4px 12px; white-space: nowrap;">X-VLA</td>
      <td align="center">72.9</td>
      <td align="center">72.8</td>
    </tr>
    <tr>
      <td style="padding: 4px 12px; white-space: nowrap;">&pi;<sub>0</sub></td>
      <td align="center">65.9</td>
      <td align="center">58.4</td>
    </tr>
    <tr>
      <td style="padding: 4px 12px; white-space: nowrap;">&pi;<sub>0.5</sub></td>
      <td align="center">82.7</td>
      <td align="center">76.8</td>
    </tr>
    <tr>
      <td style="padding: 4px 12px; white-space: nowrap;">Motus</td>
      <td align="center"><u>88.7</u></td>
      <td align="center"><u>87.0</u></td>
    </tr>
    <tr style="border-top: 1px solid black; border-bottom: 2px solid black;">
      <td style="padding: 6px 12px; white-space: nowrap;"><b>LingBot-VA (Ours)</b></td>
      <td align="center"><b>92.9</b> <small>(+4.2)</small></td>
      <td align="center"><b>91.6</b> <small>(+4.6)</small></td>
    </tr>
  </tbody>
</table>


- **LIBERO**

<table style="border-collapse: collapse; width: auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; font-size: 13px; line-height: 1.2;">
<!-- 指标说明 -->
  <p style="font-size: 12px; color: #666; margin-bottom: 5px;">* All metrics are reported in percentage (%). Higher values are <b>bolded</b>.</p>
  <thead>
    <tr style="border-top: 2px solid black; border-bottom: 1px solid black;">
      <th align="left" style="padding: 6px 10px; border-right: 1px solid black; white-space: nowrap;">Methods</th>
      <th align="center" style="padding: 6px 8px;">Spatial</th>
      <th align="center" style="padding: 6px 8px;">Object</th>
      <th align="center" style="padding: 6px 8px;">Goal</th>
      <th align="center" style="padding: 6px 8px;">Long</th>
      <th align="center" style="padding: 6px 8px;">Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 4px 10px; border-right: 1px solid black; white-space: nowrap;">&pi;<sub>0</sub></td>
      <td align="center">96.8</td><td align="center">98.8</td><td align="center">95.8</td><td align="center">85.2</td><td align="center">94.1</td>
    </tr>
    <tr>
      <td style="padding: 4px 10px; border-right: 1px solid black; white-space: nowrap;">&pi;<sub>0.5</sub></td>
      <td align="center">98.8</td><td align="center">98.2</td><td align="center">98.0</td><td align="center">92.4</td><td align="center">96.9</td>
    </tr>
    <tr>
      <td style="padding: 4px 10px; border-right: 1px solid black; white-space: nowrap;">OpenVLA</td>
      <td align="center">84.7</td><td align="center">88.4</td><td align="center">79.2</td><td align="center">53.7</td><td align="center">76.5</td>
    </tr>
    <tr>
      <td style="padding: 4px 10px; border-right: 1px solid black; white-space: nowrap;">X-VLA</td>
      <td align="center">98.2</td><td align="center">98.6</td><td align="center">97.8</td><td align="center">97.6</td><td align="center">98.1</td>
    </tr>
    <tr style="border-top: 1.5px solid black; border-bottom: 2px solid black;">
      <td style="padding: 5px 10px; border-right: 1px solid black; white-space: nowrap;"><b>LingBot-VA (Ours)</b></td>
      <td align="center"><b>98.5 &plusmn; 0.3</b></td>
      <td align="center"><b>99.6 &plusmn; 0.3</b></td>
      <td align="center"><b>97.2 &plusmn; 0.2</b></td>
      <td align="center"><b>98.5 &plusmn; 0.5</b></td>
      <td align="center"><b>98.5</b></td>
    </tr>
  </tbody>
</table>



&nbsp;

## Real-world Deployment

Six manipulation tasks across three categories: longhorizon tasks (Make Breakfast, Pick Screws), precision tasks (Insert Tube, Unpack Delivery), and deformable & articulated object
manipulation (Fold Clothes, Fold Pants). Our method achieves state-of-the-art performance on both metrics (Progress Rate and Success Rate) with <b>only 50 trials</b> per task, substantially outperforming strong baseline &pi;<sub>0.5</sub>.

<div style="text-align: left; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; line-height: 1.6;">

  <!-- 第一部分：PS 说明 -->
  <div style="margin-bottom: 5px;"><strong>Progress Score (PS):</strong> The average score across all trials divided by the maximum possible score, expressed as a percentage:</div>

  PS = Average_Progress / Max_Steps &times; 100%

  <!-- 第二部分：SR 说明 -->
  <div style="margin-bottom: 5px;"><strong>Success Rate (SR):</strong> The number of successful trials divided by the total number of trials, expressed as a percentage:</div>

  SR = Successful_Trials / N &times; 100%

</div>



<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;">
  <!-- 指标说明 -->
  <p style="font-size: 12px; color: #666; margin-bottom: 5px;">* All metrics are reported in percentage (%). Higher values are <b>bolded</b>.</p>
  
  <table style="border-collapse: collapse; width: auto; font-size: 13px; line-height: 1.2;">
    <thead>
      <tr style="border-top: 2px solid black;">
        <th rowspan="2" align="left" style="padding: 4px 10px; border-bottom: 1px solid black; white-space: nowrap;"><b>Task</b></th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">Make Breakfast</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">Pick Screws</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">Insert Tube</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">Unpack Delivery</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">Fold Clothes</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">Fold Pants</th>
      </tr>
      <tr style="border-bottom: 1px solid black;">
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding: 6px 10px; white-space: nowrap;">&pi;<sub>0.5</sub></td>
        <td align="center">73.0</td><td align="center">70.0</td>
        <td align="center">74.0</td><td align="center">50.0</td>
        <td align="center">79.2</td><td align="center">30.0</td>
        <td align="center">73.0</td><td align="center">25.0</td>
        <td align="center"><b>62.9</b></td><td align="center">30.0</td>
        <td align="center">30.0</td><td align="center">30.0</td>
      </tr>
      <tr style="border-bottom: 2px solid black;">
        <td style="padding: 6px 10px; white-space: nowrap;"><b>LingBot-VA (Ours)</b></td>
        <td align="center"><b>97.0</b></td><td align="center"><b>75.0</b></td>
        <td align="center"><b>82.5</b></td><td align="center"><b>70.0</b></td>
        <td align="center"><b>85.8</b></td><td align="center"><b>40.0</b></td>
        <td align="center"><b>84.5</b></td><td align="center"><b>65.0</b></td>
        <td align="center">48.8</td><td align="center"><b>35.0</b></td>
        <td align="center"><b>76.7</b></td><td align="center"><b>70.0</b></td>
      </tr>
    </tbody>
  </table>
</div>


# 🪪 License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE.txt) file for details.

# 📚Citation

```bibtex
@article{lingbot-va2026,
  title={Causal World Modeling for Robot Control},
  author={Li, Lin and Zhang, Qihang and Luo, Yiming and Yang, Shuai and Wang, Ruilin and Han, Fei and Yu, Mingrui and Gao, Zelin and Xue, Nan and Zhu, Xing and Shen, Yujun and Xu, Yinghao},
  journal={arXiv preprint arXiv:2601.21998},
  year={2026}
}
```

# 🧩 Acknowledgments

This work builds upon several excellent open-source projects:

- [Wan-Video](https://github.com/Wan-Video) - Vision transformer backbone
- [MoT](https://github.com/facebookresearch/Mixture-of-Transformers) - Mixture-of-Transformers architecture
- The broader open-source computer vision and robotics communities

---

For questions, discussions, or collaborations:

- **Issues**: Open an [issue](https://github.com/robbyant/lingbot-va/issues) on GitHub
- **Email**: Contact Dr. [Qihang Zhang](https://zqh0253.github.io/) (liuhuan.zqh@antgroup.com) or Dr. Lin Li (fengchang.ll@antgroup.com) 
