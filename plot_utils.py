import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from moviepy.editor import ImageSequenceClip

class RLPlotter():
    def __init__(self, logger, path, test=False):
        self.logger = logger
        self.path = path
        self.test = test

    def update_goal(self, goal):
        self.goal = goal

    def clear_plots(self, path):
        if not self.test:
            #clear_folder('episode_losses_critic', path)
            #clear_folder('episode_losses_actor', path)
            #clear_folder('episode_losses_sum', path)
            #clear_folder('episode_losses_overlapped', path)
            clear_folder('run_losses_overlapped', path)
            #clear_folder('run_losses_sum', path)
            #clear_folder('run_losses_critic', path)
            #clear_folder('run_losses_actor', path)
            clear_folder('episode_mean_rewards', path)
            
        #clear_folder('episode_actions', path)
        clear_folder('episode_paths', path)
        #clear_folder('episode_steps', path)
        clear_folder('episode_actions_polar', path)

    def plot_last_episode(self):
        if not self.test:
            #self.plot_last_episode_losses_critic()
            #self.plot_last_episode_losses_actor()
            #self.plot_last_episode_losses_sum()
            #self.plot_last_episode_losses_overlapped()
            self.plot_last_losses_overlapped()
            #self.plot_last_losses_sum()
            #self.plot_last_losses_critic()
            #self.plot_last_losses_actor()
            self.plot_last_mean_rewards()

        #self.plot_last_episode_actions()
        self.plot_last_episode_paths()
        #self.plot_last_episode_steps()
        self.plot_last_episdode_actions_polar()

    def plot_last_mean_rewards(self):
        folder = os.path.join(self.path, 'episode_mean_rewards')
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        x = np.arange(len(self.logger.episode_mean_rewards))
        plt.plot(x, self.logger.episode_mean_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Episode Mean Reward')
        plt.title('Mean Reward')
        plt.savefig(os.path.join(folder, 'episode_mean_rewards.png'))
        plt.close()

    def plot_last_losses_sum(self):
        folder = os.path.join(self.path, 'run_losses_sum')
        i = len(self.logger.episode_losses_critic)-1
        x = np.arange(len(self.logger.losses_critic))
        plt.plot(x, np.array(self.logger.losses_critic)+np.array(self.logger.losses_actor))
        plt.xlabel('Step')
        plt.ylabel('Loss Sum')
        plt.title(f'Run Losses Sum')
        plt.savefig(os.path.join(folder, f'run_losses_sum.png'))
        plt.close()

    def plot_last_losses_overlapped(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        folder = os.path.join(self.path, 'run_losses_overlapped')
        i = len(self.logger.episode_losses_critic)-1
        x = np.arange(len(self.logger.losses_critic))
        ax1.plot(x, self.logger.losses_critic, c='r', label='Critic')
        ax2.plot(x, self.logger.losses_actor, c='b', label='Actor')
        fig.legend(loc='upper right')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss Critic')
        ax2.set_ylabel('Loss Actor')
        ax1.set_title(f'Run Losses Overlapped')
        fig.savefig(os.path.join(folder, f'run_losses_overlapped.png'))
        plt.close()

    def plot_last_episode_losses_sum(self):        
        folder = os.path.join(self.path, 'episode_losses_sum')
        i = len(self.logger.episode_losses_critic)-1
        x = np.arange(len(self.logger.episode_losses_critic[i]))
        plt.plot(x, np.array(self.logger.episode_losses_critic[i])+np.array(self.logger.episode_losses_actor[i]))
        plt.xlabel('Step')
        plt.ylabel('Loss Sum')
        plt.title(f'Episode {i} MSE')
        plt.savefig(os.path.join(folder, f'episode{i:02}_losses_sum.png'))
        plt.close()

    def plot_last_episode_losses_overlapped(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        folder = os.path.join(self.path, 'episode_losses_overlapped')
        i = len(self.logger.episode_losses_critic)-1
        x = np.arange(len(self.logger.episode_losses_critic[i]))
        ax1.plot(x, self.logger.episode_losses_critic[i], c='r', label='Critic')
        ax2.plot(x, self.logger.episode_losses_actor[i], c='b', label='Actor')
        fig.legend(loc='upper right')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss Critic')
        ax2.set_ylabel('Loss Actor')
        ax1.set_title(f'Episode {i} MSE')
        fig.savefig(os.path.join(folder, f'episode{i:02}_losses_overlapped.png'))
        plt.close()

    def plot_last_episode_losses_critic(self):        
        folder = os.path.join(self.path, 'episode_losses_critic')
        i = len(self.logger.episode_losses_critic)-1
        x = np.arange(len(self.logger.episode_losses_critic[i]))
        plt.plot(x, self.logger.episode_losses_critic[i])
        plt.xlabel('Step')
        plt.ylabel('Loss Critic')
        plt.title(f'Episode {i} MSE')
        plt.savefig(os.path.join(folder, f'episode{i:02}_losses_critic.png'))
        plt.close()
    
    def plot_last_episode_losses_actor(self):
        folder = os.path.join(self.path, 'episode_losses_actor')
        i = len(self.logger.episode_losses_actor)-1
        x = np.arange(len(self.logger.episode_losses_actor[i]))
        plt.plot(x, self.logger.episode_losses_actor[i])
        plt.xlabel('Step')
        plt.ylabel('Loss Actor')
        plt.title(f'Episode {i} MSE')
        plt.savefig(os.path.join(folder, f'episode{i:02}_losses_actor.png'))
        plt.close()

    def plot_last_losses_critic(self):
        folder = os.path.join(self.path, 'run_losses_critic')
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        x = np.arange(len(self.logger.losses_critic))
        plt.plot(x, self.logger.losses_critic)
        plt.xlabel('Step')
        plt.ylabel('Loss Critic')
        plt.title('Run MSE')
        plt.savefig(os.path.join(folder, 'run_losses_critic.png'))
        plt.close()
    
    def plot_last_losses_actor(self):
        folder = os.path.join(self.path, 'run_losses_actor')
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        x = np.arange(len(self.logger.losses_actor))
        plt.plot(x, self.logger.losses_actor)
        plt.xlabel('Step')
        plt.ylabel('Loss Actor')
        plt.title('Run MSE')
        plt.savefig(os.path.join(folder, 'run_losses_actor.png'))
        plt.close()

    def plot_last_episode_steps(self):
        folder = os.path.join(self.path, 'episode_steps')
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        x = np.arange(len(self.logger.episode_steps))
        plt.plot(x, self.logger.episode_steps)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.savefig(os.path.join(folder, 'episode_steps.png'))
        plt.close()
    
    def plot_last_episode_actions(self):
        folder = os.path.join(self.path, 'episode_actions')
        i = len(self.logger.episode_actions) - 1
        x = np.arange(len(self.logger.episode_actions[i]))
        plt.plot(x, self.logger.episode_actions[i])
        plt.xlabel('Step')
        plt.ylabel(r'Active Orientation $\theta$')
        plt.ylim(0,7)
        plt.savefig(os.path.join(folder, f'episode{i:02}_actions.png'))
        plt.close()
    
    def plot_last_episdode_actions_polar(self):
        folder = os.path.join(self.path, 'episode_actions_polar')
        i = len(self.logger.episode_actions) - 1
        x = np.arange(len(self.logger.episode_actions[i]))
        fig = plt.figure()
        ax = plt.subplot(111, polar=True)
        ax.scatter(self.logger.episode_actions[i], x, c=x, cmap = 'plasma')
        fig.savefig(os.path.join(folder, f'episode{i:02}_actions_polar.png'))
        plt.close()

    def plot_last_episode_paths(self):
        folder = os.path.join(self.path, 'episode_paths')
        i = len(self.logger.episode_states) - 1
        x = np.array(self.logger.episode_states[i])[:,:,0]
        y = np.array(self.logger.episode_states[i])[:,:,1]
        plot_normalized_concentration(self.goal)
        colormap_array = np.linspace(0,1,x.shape[0])
        for j in range(x.shape[1]):
            plt.scatter(x[:,j], y[:,j], c=colormap_array, cmap='RdBu', alpha=0.5)
        # plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')

        plt.title(f'Episode {i} Path')
        plt.savefig(os.path.join(folder, f'episode{i:02}_path.png'))
        plt.close()

def plot_normalized_concentration(goal, show=False):
    x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))

    r = np.sqrt((x-goal.center[0][0])**2+(y-goal.center[0][1])**2)

    concentration = 1/(1+r)

    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.imshow(concentration,cmap = 'plasma',extent=[-1,1,-1,1],origin='lower')
    colorbar = plt.colorbar()
    colorbar.set_label(r'$c/c_0$',labelpad=10,fontsize = 20)
    if show:
        plt.show()

def clear_folder(folder_name, path):
    folder = os.path.join(path, folder_name)
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)

def make_animation(path, fps):
    folder_path = path  # update this with the path to the folder containing your images
    image_files = sorted([os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(".png")])
    clip = ImageSequenceClip(image_files, fps)  # adjust fps (frames per second) as needed
    clip.write_videofile(str(path)+"_animation.mp4")  # save the animation to a file