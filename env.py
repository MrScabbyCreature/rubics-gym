import gym
from gym import spaces
from cube import Cube, num_to_face_map
import numpy as np

class CubeEnv(gym.Env):
	''' Rubics cube gym environment '''
	metadata = {'render.modes': ['term', 'gui']}

	def __init__(self, n):
		super().__init__()
		self.n = n
		self.action_space = spaces.Tuple(
									(
										spaces.Discrete(6), # faces
										spaces.Discrete(2), # direction - clockwise/anticlockwise
										spaces.Discrete((self.n - 1) // 2), #slice distance from face
									)
									)
		self.cube = None
		self.reset()

	def step(self, action):
		# Execute one time step within the environment
		self.cube.rotate(*action)
		face_order_for_obs = ['F', 'L', 'B', 'R', 'U', 'D']
		obs = np.stack([self.cube.faces[face][...] for face in face_order_for_obs], axis=2)
		reward = 0 # FIXME
		done = self.cube.is_complete()
		info = {}
		return obs, reward, done, info

	def reset(self):
		# Reset the state of the environment to an initial state
		del self.cube
		self.cube = Cube(self.n)
		for i in range(self.n * 10):
			self.step(self.action_space.sample())

	def render(self, mode='term'):
		# Render the environment to the screen
		render_modes = ['term', 'gui']
		assert mode in render_modes, f"Possible modes: {render_modes}. Passed: {mode}"
		if mode == 'term':
			self.cube.print_cube_with_colors()
		else:
			self.cube.plot_cube()

env = CubeEnv(n=3)
env.render("gui")
import time
time.sleep(2)
env.close()
