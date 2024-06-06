<p align="center">
  <a href="#">
    <img src="https://github.com/Apocaypse/roboMamba/blob/main/assets/structure.png?raw=true">
  </a>
</p>

# Robot-Mamba: Multimodal State Space Model for Efficient Robot Reasoning and Manipulation
The repo of paper `RoboMamba: Multimodal State Space Model for Efficient Robot Reasoning and Manipulation`

The contribution of our work can be broadly divided into three parts：

- First of all, We innovatively integrate a vision encoder with the efficient Mamba language model to construct our end-to-end RoboMamba, which possesses visual common sense and robot-related reasoning abilities.
- To equip RoboMamba with action pose prediction abilities, we explore an efficient fine-tuning strategy using a simple policy head. We find that once RoboMamba achieves sufficient reasoning capabilities, it can acquire pose prediction skills with minimal cost.
- In our extensive experiments, RoboMamba excels in reasoning on general and robotic evaluation benchmarks, and showcases impressive pose prediction results in both simulation and real-world experiments. Our model has better performance, lower cost, and faster inference time.

## Demo Video
https://github.com/Apocaypse/roboMamba/blob/main/assets/720p.mp4

## Usage

<a href="#">
    <img src="https://github.com/Apocaypse/roboMamba/blob/main/assets/usage.png?raw=true">
</a>