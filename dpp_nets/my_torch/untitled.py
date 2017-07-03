class DPP_Regressor(nn.Module):
    
    def __init__(self, network_params, train_params, data_params):
        
        super(DPP_Regressor, self).__init__()
        
        self.dtype = torch.DoubleTensor
        self.dim_in = network_params['dim_in']
        self.dim_hk = network_params['dim_hk']
        self.dim_kern= network_params['dim_kern']
        self.dim_h1 = network_params['dim_h1']
        self.reg_kern = train_params['reg_kern']
        self.lr = train_params['lr']
        self.batch_size = train_params['batch_size']
        self.probs = data_params['probs']
        self.n_cluster = self.probs.size(0)
        self.dim_emb = data_params['dim_emb']
        self.set_size = data_params['set_size']
        self.signal_scale = data_params['signal_scale']
        self.signal_bias = data_params['signal_bias']

        self.emb_layer = torch.nn.Sequential(nn.Linear(self.dim_in,self.dim_hk),nn.ELU(),
                                             nn.Linear(self.dim_hk,self.dim_kern))
        self.dpp_layer = DPP_Layer()
        self.prediction_layer = torch.nn.Sequential(nn.Linear(int(self.dim_in / 2),self.dim_h1),nn.ReLU(),
                                                    nn.Linear(self.dim_h1,1))
                
        self.criterion = nn.MSELoss()

        self.embedding = Variable(torch.randn(self.set_size, self.dim_kern))
        self.subset = Variable(torch.randn(1,1))
        self.filtered_and_summed_x = Variable(torch.randn(1,1))
        self.alpha = 0
        
        self.grad_dict = defaultdict(list)
        self.mod_grad_dict = defaultdict(list)
        self.loss_dict = defaultdict(list)
        self.step_dict = defaultdict(list)
        self.super_mod_grad_dict = defaultdict(list)
        
        self.score_dict = defaultdict(list) # saves a list of sampled scores at time t
        self.score_mean = dict()
        self.score_var = dict()
        self.score_coef = dict()
        self.reinforce_dict = defaultdict(list) # saves a list of sampled REINFORCE grads at time t
        self.reinforce_mean = dict()
        self.reinforce_var = dict()
        self.reinforce_coef = dict()
        self.loss_dict = defaultdict(list) # saves a list of sampled losses at time t
        self.loss_mean = dict()
        self.loss_var = dict()
        self.loss_coef = dict()
        self.mod_dict = defaultdict(list) # saves a list of the modified grads at time t
        self.mod_mean = dict()
        self.mod_var = dict()
        self.mod_coef = dict()
        
        self.weight_norm_1 = dict()
        self.weight_max_1 = dict()
        self.weight_norm_2 = dict()
        self.weight_max_2 = dict()
        
        # does the same for the alpha part
        self.a_score_dict = defaultdict(list)
        self.a_reinforce_dict = defaultdict(list)
        self.a_loss_dict = defaultdict(list)
        self.a_mod_dict = defaultdict(list)
        self.alpha_dict = defaultdict(list)
        
        self.x_dict = defaultdict(list)
        self.y_dict = defaultdict(list)
        self.emb_dict = defaultdict(list)
        self.subset_dict = defaultdict(list)
        self.y_pred_dict = defaultdict(list)
        self.summed_dict = defaultdict(list)
        
        self.grad_mean = {}
        self.mod_grad_mean = {}
        self.loss_mean = {}
        self.step_mean = {}
        self.super_mod_grad_mean = {}
        
        self.grad_var = {}
        self.mod_grad_var = {}
        self.loss_var = {}
        self.step_var = {}
        self.super_mod_grad_var = {}
        self.alphas = defaultdict(list)
        
        self.double()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim = 1)
        self.embedding = self.emb_layer(x)
        self.subset = torch.diag(self.dpp_layer(self.embedding).type(self.dtype))
        self.filtered_and_summed_x = self.subset.mm(x1).sum(0)
        pred = self.prediction_layer(self.filtered_and_summed_x)
        return pred
      
    def gen_data(self):
        stuck = 5

        x1 = np.random.randn(self.set_size, self.dim_in // 2)

        clusters = np.random.choice(self.dim_in // (2*stuck), np.random.choice(20,1) + 1, replace=False)
        n = len(clusters)
        rep = (40 // n) + 1
        clusters = np.array([np.arange(i*5, i*5 + 5) for i in clusters]).flatten()
        clusters = np.tile(clusters, rep)[:(self.set_size * stuck)]
        x1[np.repeat(np.arange(40), stuck), clusters] = np.random.normal(50, 1, (self.set_size * stuck))
        np.random.shuffle(x1)
        x2 = Variable(torch.Tensor(np.tile(np.sum(x1, axis=0), (self.set_size, 1)))).double()
        x1 = Variable(torch.Tensor(x1)).double()
        #x = Variable(torch.Tensor(np.hstack([x1, x2])))
        y = Variable(torch.Tensor([n])).double()
        ind = clusters
        
        return x1, x2, y, ind
    
    def sample(self):
        x1, x2, y, ind = self.gen_data()
        y_pred = self.forward(x1, x2)
        
     
        print("Subset Size: ", self.subset.sum().data[0])
        print("Number of True Clusters is: ", y.data[0])
        print("Prediction is: ", y_pred.data[0])
        
        return y_pred, y, x1, x2
    
    def train(self, train_iter, batch_size, lr, reg):
        
        self.lr = lr
        self.batch_size = batch_size
        
        self.optimizer = optim.SGD([{'params': self.emb_layer.parameters(), 'weight_decay': self.reg_kern},
                                    {'params': self.prediction_layer.parameters()}],
                                    lr = self.lr / self.batch_size)

        for t in range(train_iter):
            x1, x2, y, _ = self.gen_data()
            #print("x is", x,"y is", y)
            #self.x_dict[t].append(x.data)
            #self.y_dict[t].append(y.data)
            
            mod_grad = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: (grad_in[0] * loss.data[0],))
            y_pred = self.forward(x1, x2)
            
            self.y_pred_dict[t].append(y_pred.data)
            self.emb_dict[t].append(self.embedding.data)
            self.subset_dict[t].append(self.subset.data)
            self.summed_dict[t].append(self.filtered_and_summed_x.data)

            loss = self.criterion(y_pred,y)
            #print()
            self.loss_dict[t].append(loss.data)
            reg_loss = loss + reg * torch.pow(torch.trace(self.embedding),2)
            
            reg_loss.backward()

            mod_grad.remove()
            
            if (t + 1) % self.batch_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if (t + 1) % 50 == 0:
                print(t + 1, loss.data[0])
                
            #exp_lr_scheduler(self.optimizer, t, lr_decay=lr_decay, lr_decay_step=ev_step)
            
    def train_with_baseline(self, train_iter, batch_size, sample_iter, alpha_iter, lr, aggregate=False):
        
        # Prepare Optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optim.SGD([{'params': self.emb_layer.parameters(), 'weight_decay': self.reg_kern * 
                                    self.batch_size * sample_iter},
                                    {'params': self.prediction_layer.parameters()}],
                                    lr = self.lr / (self.batch_size * sample_iter))
        
        # Clear dictionaries
        self.score_dict.clear()
        self.score_mean.clear()
        self.score_var.clear()
        self.reinforce_dict.clear()
        self.reinforce_mean.clear()
        self.reinforce_var.clear()
        self.loss_dict.clear()
        self.loss_mean.clear()
        self.loss_var.clear()
        self.mod_dict.clear()
        self.mod_mean.clear()
        self.mod_var.clear()
        
        for t in range(train_iter):
            
            self.weight_norm_1[t] = torch.mean(torch.pow(self.emb_layer[0].weight.data,2))
            self.weight_max_1[t] = torch.max(torch.pow(self.emb_layer[0].weight.data,2))
            self.weight_norm_2[t] = torch.mean(torch.pow(self.emb_layer[2].weight.data,2))
            self.weight_max_2[t] = torch.max(torch.pow(self.emb_layer[2].weight.data,2))

            # Draw a Training Sample
            x1, x2, y, _ = self.gen_data()
            
            # Save score, build reinforce gradient, save reinforce gradient
            save_score = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.a_score_dict[t].append(grad_in[0].data))
            reinforce_grad = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: (grad_in[0] * (loss.data[0]),))
            save_reinforce = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.a_reinforce_dict[t].append(grad_in[0].data))
            
            # Estimate alpha
            if alpha_iter:
                for i in range(alpha_iter):                                      
                    y_pred = self.forward(x1, x2)
                    loss = self.criterion(y_pred, y)
                    loss.backward()

                self.alpha = compute_alpha(self.a_reinforce_dict[t], self.a_score_dict[t], False, True, False).double()
                self.zero_grad()
                self.alphas[t].append(self.alpha)
                
            else:
                self.alpha = torch.zeros(self.embedding.size()).double()
                #self.alpha = 0
                    
            save_score.remove()
            reinforce_grad.remove()
            save_reinforce.remove()
            
            # now actual training
            # save scores, reinforce, implement baseline gradient, save baseline gradient
            save_score = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.score_dict[t].append(grad_in[0].data))
            save_reinforce = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.reinforce_dict[t].append(grad_in[0].data * loss.data[0]))
            modify_grad = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: (Variable(grad_in[0].data * (loss.data[0] - self.alpha)),))
            save_modified = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.mod_dict[t].append(grad_in[0].data))

            # sample multiple times from the DPP and backpropagate associated gradients!
            for i in range(sample_iter):
                y_pred = self.forward(x1, x2)
                loss = self.criterion(y_pred, y)
                self.loss_dict[t].append(loss.data) # save_loss
                loss.backward()

            # update parameters after processing a batch
            if (t + 1) % self.batch_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # print loss
            if (t + 1) % 50 == 0:
                print(t + 1, loss.data[0])
                
            save_score.remove()
            save_reinforce.remove()
            modify_grad.remove()
            save_modified.remove()

            
            # anneal learning rate
            #exp_lr_scheduler(self.optimizer, t, lr_decay=lr_decay, lr_decay_step=ev_step)
                
        # Update Dictionaries
        self.score_mean = {k: torch.mean(torch.stack(v)) for k, v in self.score_dict.items()}
        self.score_var  = {k: torch.mean(torch.var(torch.stack(v), dim=0)) for k, v in self.score_dict.items()}
        self.score_coef = {k: torch.mean(torch.std(torch.stack(v), dim=0) / torch.mean(torch.stack(v), dim=0))
                          for k, v in self.score_dict.items()}

        self.reinforce_mean = {k: torch.mean(torch.stack(v)) for k, v in self.reinforce_dict.items()}
        self.reinforce_var  = {k: torch.mean(torch.var(torch.stack(v), dim=0)) for k, v in self.reinforce_dict.items()}
        self.reinforce_coef = {k: torch.mean(torch.std(torch.stack(v), dim=0) / torch.mean(torch.stack(v), dim=0))
                          for k, v in self.reinforce_dict.items()}

        self.loss_mean = {k: torch.mean(torch.stack(v)) for k, v in self.loss_dict.items()}
        self.loss_var  = {k: torch.mean(torch.var(torch.stack(v), dim=0)) for k, v in self.loss_dict.items()}
        self.loss_coef = {k: torch.mean(torch.std(torch.stack(v), dim=0) / torch.mean(torch.stack(v), dim=0))
                          for k, v in self.loss_coef.items()}

        self.mod_mean = {k: torch.mean(torch.stack(v)) for k, v in self.mod_dict.items()}
        self.mod_var  = {k: torch.mean(torch.var(torch.stack(v), dim=0)) for k, v in self.mod_dict.items()}
        self.mod_coef = {k: torch.mean(torch.std(torch.stack(v), dim=0) / torch.mean(torch.stack(v), dim=0))
                          for k, v in self.mod_dict.items()}

                        
    def evaluate(self,test_iter):

        loss = 0.0
        subset_count = 0.0
        
        for t in range(test_iter):
            x1, x2, y, ind = self.gen_data()
            y_pred = self.forward(x1, x2)
            loss += self.criterion(y_pred,y)
            subset_count += torch.sum(self.subset.data)
            
        print("Average Loss is: ", loss / test_iter) 
        print("Average Subset Size is: ", subset_count / (test_iter))
        
    def reset_parameter(self):
        self.emb_layer[0].reset_parameters()
        self.emb_layer[2].reset_parameters()
        self.prediction_layer[0].reset_parameters()
        self.prediction_layer[2].reset_parameters()
        
        self.optimizer = optim.SGD([{'params': self.emb_layer.parameters(), 'weight_decay': self.reg_kern},
                                    {'params': self.prediction_layer.parameters()}],
                                    lr = self.lr / self.batch_size)
