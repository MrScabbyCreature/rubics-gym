import pickle
from env import CubeEnv

env = CubeEnv(3)
start_state = env.reset(randomize=False)

max_states = 1000000000000000
completed_states = 0
repeated_states = 0
state_to_action = {}
state_queue = [start_state]

pickle_checkpoint = 0
pickle_increment = 10**5

while completed_states <= max_states:
    current_state = state_queue.pop(0)
    for face in range(6):
        for direction in range(2):
            env.reset(state=current_state)
            action = [face, direction, 0]
            new_state, r, d, _ = env.step(action)
            new_state_bytes = new_state.tobytes()
            if new_state_bytes not in state_to_action:
                state_queue.append(new_state)
                reverse_action = [action[0], abs(action[1] - 1), action[2]]
                state_to_action[new_state_bytes] = reverse_action
                completed_states += 1
            else:
                repeated_states += 1
    print(completed_states, repeated_states)

    if pickle_checkpoint + pickle_increment < completed_states:
        pickle.dump(state_to_action, open(f"data/{completed_states}.pickle", 'wb'))
        pickle_checkpoint += pickle_increment
        print(f"Pickled data/{completed_states}.pickle")
