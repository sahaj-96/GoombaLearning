# GoombaLearning
## Introduction
A **Reinforcement Learning** project aimed at developing AI bot capable of playing **Super Mario Bros**, utilizing **Proximal Policy Optimization**.


https://github.com/sahaj-96/GoombaLearning/assets/113182228/aea3bb2b-302b-449e-a562-cd802a19e2fa

### Installation and Dependencies
- OpenCV
- OpenAI Gym
- Super Mario Bros Environment developed by [Kautenja](https://github.com/Kautenja/gym-super-mario-bros)
  
All requirements have been clearly mentioned in the **requirements.txt** <br>
Just do the following:-

<pre>
<code class="language-bash">
python -m venv env_name <span style="color: green">                 #Create a virtual env named env_name</span>
env_name\Scripts\activate 
git clone https://github.com/sahaj-96/GoombaLearning.git 
cd GoombaLearning
pip install -r Ignision/requirements.txt <span style="color: green">#Download the required libraries</span>
</code>
</pre>
**Note:** *This is only applicable to the Windows Terminal.*

### Training
<pre>
<code class="language-bash">
python -m GoombaLearning --train --epochs=10<span style="color: green"></span>
</code>
</pre>
### Testing
<pre>
<code class="language-bash">
python -m GoombaLearning --test --episodes=10<span style="color: green"></span>
</code>
</pre>
**Note:** *These parameters are merely exemplary adjust them according to your requirements.*
## Video Demonstration
[![Follow this link https://www.youtube.com/watch?v=RmCMADcMDx8 ](http://img.youtube.com/vi/RmCMADcMDx8/0.jpg)](https://www.youtube.com/watch?v=RmCMADcMDx8)

## References
[ Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)

[Super Mario Bros Environment](https://github.com/Kautenja/gym-super-mario-bros)
## Contributors
[Sahaj Srivastava](https://github.com/sahaj-96) <br>



