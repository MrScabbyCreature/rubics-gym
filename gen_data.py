import pickle
from env import CubeEnv

env = CubeEnv(3)
start_state = env.reset(randomize=False)

# variables for constant monitoring
completed_states = 0
repeated_states = 0

# lists and dict to store data
state_to_action = {}
state_queue = []
next_state_queue = [start_state]

# for checkpointing to pickle
pickle_checkpoint = 0
pickle_increment = 10**5

# list of possible actions
action_list = []
for face in range(6):
    for direction in range(2):
        action_list.append([face, direction, 0])

print_tracker = 0
# loop 
for steps_from_finished in range(1, 10):
    print("\n\nSteps from finished:", steps_from_finished)
    state_queue = next_state_queue
    next_state_queue = []
    for current_state in state_queue:
        for action in action_list:
            env.reset(state=current_state)
            new_state, r, d, _ = env.step(action)
            new_state_bytes = new_state.tobytes()
            if new_state_bytes not in state_to_action:
                next_state_queue.append(new_state)
                reverse_action = [action[0], abs(action[1] - 1), action[2]]
                state_to_action[new_state_bytes] = reverse_action
                completed_states += 1
            else:
                repeated_states += 1
        if print_tracker == 100:
            print(completed_states, repeated_states, end='\r')
            print_tracker = 0
        print_tracker += 1

        if pickle_checkpoint + pickle_increment < completed_states:
            pickle.dump(state_to_action, open(f"data/{completed_states}.pickle", 'wb'))
            pickle_checkpoint += pickle_increment
            print(f"\t\t\t\tPickled data/{completed_states}.pickle")
    print(completed_states, repeated_states, end='\r')
