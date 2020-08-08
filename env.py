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
		self.face_order_for_obs = ['F', 'L', 'B', 'R', 'U', 'D']
		self.cube = None
		self.reset()

	def step(self, action):
		# Execute one time step within the environment
		self.cube.rotate(*action)
		obs = np.stack([self.cube.faces[face][...] for face in self.face_order_for_obs], axis=2)
		reward = 0 # TODO: To experiment
		done = self.cube.is_complete()
		info = {}
		return obs, reward, done, info

	def reset(self, randomize=True, state=None):
		'''Reset the env
		args
		----
		randomize(optional): Bool
			If True, returns a random state. If False, returns a completed cube. True by default
		state(optiona): 3x3x6 np.array representing a valid cube state
			If provided, then used as starting state. Overrides 'randomize'
		'''
		del self.cube
		self.cube = Cube(self.n)
		if randomize and isinstance(state, type(None)):
			for i in range(self.n * 10):
				self.step(self.action_space.sample())
		
		if state is not None:
			assert state.shape == (3, 3, 6), f"{state.shape} is incorrect shape."
			for i in range(6):
				self.cube.faces[self.face_order_for_obs[i]][...] = state[:, :, i]
				
		obs = np.stack([self.cube.faces[face][...] for face in self.face_order_for_obs], axis=2)
		return obs

	def render(self, mode='gui'):
		# Render the environment to the screen
		render_modes = ['term', 'gui']
		assert mode in render_modes, f"Possible modes: {render_modes}. Passed: {mode}"
		if mode == 'term':
			self.cube.print_cube_with_colors()
		else:
			self.cube.plot_cube()

if __name__ == "__main__":
	import time

	env = CubeEnv(n=3)
	env.reset(randomize=False)
	for i in range(10):
		action = env.action_space.sample()
		print("Action", action)
		env.step(action)
		env.render("gui")
		input()
	env.close()
