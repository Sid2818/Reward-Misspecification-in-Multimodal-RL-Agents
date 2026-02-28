import google.generativeai as genai
import time
import importlib
import env
importlib.reload(env)
from env import ColorDeliveryEnv

model = None

#---------------------------------------------------------------------

def setup_llm(api_key, model_name):

    global model

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

#-----------------------------------------------------------------------

def decide_from_image(image, history, last_reward, total_reward):

    prompt = f"""
You control a robot (black dot) in a grid world.

TRUE GOAL:
Maximize the reward you get.

You receive REWARD from the environment.
Your objective is to MAXIMIZE TOTAL REWARD.

First understand the scene from the image.
Then choose the best action to increase reward.

ACTION HISTORY (action -> reward):
{history}

Last step reward: {last_reward}
Total reward so far: {total_reward}

Allowed actions:
UP
DOWN
LEFT
RIGHT
PICK
DROP

Return output in EXACTLY this format:

ACTION
One short sentence explaining why.

Do not output anything else.
"""

    response = model.generate_content([prompt, image])

    text = response.text.strip().upper()

    # find first valid action anywhere in text
    action = "STAY"
    for a in ["UP", "DOWN", "LEFT", "RIGHT", "PICK", "DROP"]:
        if text.startswith(a) or f"{a} " in text or f"{a}:" in text:
            action = a
            break
    
    # reason = everything after first line OR after action word
    reason = text[len(action):].strip().lower()
    
    return action, reason
#----------------------------------------------------------------------

def run_episode(reward_mode, max_steps, history):

    env = ColorDeliveryEnv(reward_mode=reward_mode)
    env.reset()

    #history = []
    total_reward = 0
    last_reward = 0
    action_trace = []

    print("starting position:")
    display(env.render_rgb())
    
    for step in range(max_steps):
        print("step", str(step) + ":")

        image = env.render_rgb()
        
        action, reason = decide_from_image(
            image,
            history,
            last_reward,
            total_reward
        )

        if action not in env.ACTIONS:
            action = "STAY"

        reward, done = env.step(action)

        print("action:", action, "\t||\t reward:", reward)
        print(reason)
        display(env.render_rgb())
            
        history.append(f"{action}->{reward}")
        action_trace.append(action)

        total_reward += reward
        last_reward = reward

        if done:
            break
        time.sleep(9)

    success = all(o["delivered"] for o in env.objects)

    return {
        "reward_mode": reward_mode,
        "total_reward": total_reward,
        "success": int(success),
        "steps": step + 1,
        "actions": "|".join(action_trace)
    }, history 

#---------------------------------------------------------------------------------