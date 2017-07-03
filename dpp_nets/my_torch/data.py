    def gen_data(self):
        """ 
        Data for regression
        """
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

# create an object that can return a single training instance (for sample) or a batch of training instances 
# for training and testing