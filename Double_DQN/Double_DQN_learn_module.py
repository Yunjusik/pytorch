def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool) ## ex_) 256

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

###################### standard - DQN ###########################
  #  next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()


###################### Standard DQN end ############################

################Dueling DQN V1###############
#    a_prime = policy_net(non_final_next_states).max(1)[1].detach() ## choose the action with maximum Q-value from policy net
#    target_dnn_output = target_net(non_final_next_states).detach() ## output of target net when inserting next_state batch
#    final_value = target_dnn_output.gather(1, a_prime.unsqueeze(1)) ## output of target net when inserting action with a_prime
#    next_state_values[non_final_mask] = final_value.squeeze() ## size reshape
##############End of Dueling DQN V1##########'''


################ Dueling DQN V2 - one line ##################

    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, policy_net(non_final_next_states).max(1)[1].unsqueeze(1).detach()).squeeze(1).detach()

############### Dueling DQN V2 ##################

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
