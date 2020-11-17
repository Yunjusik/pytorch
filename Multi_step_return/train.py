
num_episodes = 2000
for i_episode in range(num_episodes):
    state = get_screen(env.reset())    #

    for t in count():
        # Select and perform an action
        action = select_action(state)             
        Next_pre_state, reward, done, const,_ = env.step(action.item())  
        reward = torch.tensor([reward], device=device)
        Total_reward.append(reward)
        print(reward, const)
        
        if not done:
            next_state = get_screen(Next_pre_state)
            memory.push(state, action, next_state, reward)
        else: # if done
            next_state = None

        
        state = next_state
        optimize_model()

        if done:
            memory.finish_nstep()
            break

        plot_durations()

    if i_episode % TARGET_UPDATE == 0:  
       target_net.load_state_dict(policy_net.state_dict())
