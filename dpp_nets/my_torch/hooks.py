def save_grads(model_dict, t): 
	"""
	A closure to save the gradient wrt to input of a nn module.
	Arguments:
	- model_dict: defaultdict(list)
	- t: dictionary key (usually training iteration)
	"""
    def hook(module, grad_input, grad_output):
        model_dict[t].append(grad_input[0].data)

    return hook

def reinforce_grad(loss):
	"""
	A closure to modify the gradient of a nn module. 
	Use to implement REINFORCE gradient. Gradients will
	be multiplied by loss.
	Arguments: 
	- loss: Gradients are multiplied by loss, should be a scalar
	"""
    def hook(module, grad_input, grad_output):
        new_grad = grad_input * loss
        return new_grad
        
    return hook