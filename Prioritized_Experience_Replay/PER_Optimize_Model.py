
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    #transitions = memory.sample(BATCH_SIZE)
    transitions, indices, weights = memory.sample(BATCH_SIZE)
    weights = torch.from_numpy(weights).to(device)
    batch = Transition(*zip(*transitions)) ### 여러개 transition을 하나로 압축시켜주는 과정
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
    loss2 = (state_action_values - expected_state_action_values.unsqueeze(1)).pow(2) * weights.unsqueeze(1) # batch x 1
    print('weight',  weights.shape, weights)
    print('indices', indices.shape, indices)
    print('indicesargmax :', np.argmax(memory.priorities),'max loss', memory.priorities[np.argmax(memory.priorities)] )
    prios = loss2 + 1e-5
    loss = loss2.mean()
    optimizer.zero_grad()
    loss.backward()
    print('priority info (BU)', memory.priorities)  # memory size, 1D
    memory.update_priorities(indices, prios.data.cpu().numpy())
   # print('priority info (AU)',memory.priorities) #memory size, 1D

  #  print('priority.shape', memory.priorities.shape) # 10000, numpy
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
