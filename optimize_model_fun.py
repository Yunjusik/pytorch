
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    #transitions = memory.sample(BATCH_SIZE)
    transitions, indices, weights = memory.sample(BATCH_SIZE)

    print(transitions.shape)
    weights = torch.from_numpy(weights).to(device)

    batch = Transition(*zip(*transitions)) ### 여러개 transition을 하나로 압축시켜주는 과정
# ### __new__() takes 5 positional arguements but 129 were given

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    ##
    #print(weights.shape)
    #print(weights) #size [128]
    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    loss2 = (state_action_values - expected_state_action_values.unsqueeze(1)).pow(2) * weights.unsqueeze(1)
    #print('weights.shape',weights.shape)

    #print('state_action_values',state_action_values.shape)
    #print('state-ex',(state_action_values - expected_state_action_values.unsqueeze(1)).pow(2).shape)
    #print('expected_state_Action_values',expected_state_action_values.shape)
    prios = loss2 + 1e-5
    #print('loss',loss)
    #print('loss.shape' , loss.shape)
    print('prios.shape',prios.shape)
    loss = loss2.mean()
    print('loss2',loss2)
    print('indices', indices.shape)
    #print('memoryinput',prios.data.squeeze(1).cpu().numpy().shape)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # indices -> 1D, prios.data.cpu().numpy() -> 2D
    memory.update_priorities(indices, prios.data.cpu().numpy())
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
