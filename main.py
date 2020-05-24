class GWAS(): 
    
    plt.style.use("dark_background")
    def __init__(self):
        self.phenotypes = False
        self.genotypes = False
        self.perc_HZ = False
        self.n_inds = 1000
        self.n_markers = 2000
        self.ids = [i for i in range(self.n_inds)]
        
    def generatePhenotypes(self):
        pheno1 = ["Pop1","Pop2","Pop3","Pop4"]
        self.phenotypes = pd.DataFrame()
        self.phenotypes["Pheno1"] = np.random.choice(pheno1,self.n_inds).tolist()
        self.phenotypes["Pheno2"] = np.random.uniform(0,8000,self.n_inds)
    
    def generateGenotypes(self): 
        bases = ["A","T","G","C",None]
        weights = [0.25,0.2,0.1,0.25,0.05]
        self.genotypes = pd.DataFrame()
        for i in range(self.n_markers):
            self.genotypes["Marker{}".format(i+1)] = random.choices(
                population = bases,weights = weights,k = self.n_inds)
        self.perc_HZ = np.random.uniform(0.01,0.9,self.n_inds).tolist()
        print(self.genotypes.head())
    
    def visualizePhenotypes(self): 
        f = plt.figure(figsize=(15,7))
        ax = f.add_subplot(121)
        ax1 = f.add_subplot(122)
        ax.hist(self.phenotypes["Pheno1"])
        ax1.hist(self.phenotypes["Pheno2"])
        ax.set_title("Pheno1",fontsize=30)
        ax1.set_title("Pheno2",fontsize=30)
        
    def checkNormality(self,mode="log"):
        if mode=="log":
            self.phenotypes["Pheno2"]=np.log(self.phenotypes["Pheno2"]+1)
            GWAS.visualizePhenotypes(self)
    
    def checkGenotypeSparsity(self): 
        f = plt.figure(figsize=(15,7))
        ax = f.add_subplot(131)
        ax1 = f.add_subplot(132)
        ax2 = f.add_subplot(133)
        marker_sparsity = [sum([1 if i == None else 0 for i in self.genotypes.iloc[:,j]]) 
                        for j in range(self.n_markers)]
        ind_sparsity = [sum([1 if i == None else 0 for i in self.genotypes.iloc[j,:]]) 
                        for j in range(self.n_inds)]
        marker_sparsity = [i/self.n_markers for i in marker_sparsity]
        ind_sparsity = [i/self.n_inds for i in ind_sparsity]
        ax.hist(marker_sparsity)
        ax.set_title("Marker sparsity",fontsize=30)
        ax1.hist(ind_sparsity)
        ax1.set_title("Individual sparsity",fontsize=30)
        ax2.hist(self.perc_HZ)
        ax2.set_title("% Heterozygotes",fontsize=30)
        self.marker_sparsity = marker_sparsity
        self.ind_sparsity = ind_sparsity
        
    def filterGenotypesSparsity(self,thresh_ind=0.12,thresh_mark=0.03):
        #rint(self.ind_sparsity)
        target_inds = [idx for idx,i in enumerate(self.ind_sparsity) if i<thresh_ind]
        target_markers = [idx for idx,i in enumerate(self.marker_sparsity) if i<thresh_mark]
        #rint(len(target_inds))
        self.genotypes = self.genotypes.iloc[target_inds,target_markers]
        f = plt.figure(figsize=(10,10))
        x = [i for i in range(2)]
        values = [len(target_inds),self.n_inds,len(target_markers),self.n_markers]
        print("Kept individuals = {}/{}".format(values[0],values[1]))
        print("Kept markers = {}/{}".format(values[2],values[3]))
        self.n_inds = self.genotypes.shape[0]
        self.n_markers = self.genotypes.shape[1]
        self.ids = [self.ids[i] for i in target_inds]
        
    def filterMAF(self,maf=0.1):
        markers_to_keep = []
        mafs = []
        for i in range(self.genotypes.shape[1]):
            freqs = Counter(self.genotypes.iloc[:,i])
            freqs = [freqs[i]/self.n_inds for i in freqs if i!=None]
            mafs.append(min(freqs))
            if min(freqs)>maf:
                markers_to_keep.append(i)
        print("Kept markers = {}/{}".format(len(markers_to_keep),self.n_markers))
        self.genotypes = self.genotypes.iloc[:,markers_to_keep]
        f = plt.figure(figsize=(10,10))
        ax = f.add_subplot(111)
        ax.hist(mafs)
        ax.set_title("Minor Allele Frequency (MAF)",fontsize=30)
        self.n_markers = self.genotypes.shape[1]
        
    def imputeMissing(self,mode="Multivariate"):
        # Genotype to integers
        g2int = {"A":1,"T":2,"G":3,"C":4,None:None}
        int2g = {1:"A",2:"T",3:"G",4:"C"}

        colnames = self.genotypes.columns.tolist()
        for markers in range(self.n_markers):
            self.genotypes.iloc[:,markers] = [g2int[self.genotypes.iloc[i,markers]] 
                                              for i in range(self.n_inds)]
        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        imputed = imputer.fit_transform(self.genotypes)
        self.genotypes = pd.DataFrame(np.round(imputed))
        self.genotypes.columns = colnames
        self.int_genotypes = self.genotypes.copy()
        for markers in range(self.n_markers):
            self.genotypes.iloc[:,markers] = [int2g[self.genotypes.iloc[i,markers]] 
                                              for i in range(self.n_inds)]
     
    def analyzePopulationStructure(self): 
        # subset phenotypes: 
        pheno_data = self.phenotypes.iloc[self.ids,:]
        
        groups = pheno_data["Pheno1"].tolist()
        # convert labels to integers 
        l2int = {"Pop1":1,"Pop2":2,"Pop3":3,"Pop4":4}
        pheno_data["Pheno1"] = [l2int[groups[i]]
                                for i in range(len(groups)) ]
        # Merge phenotypes and genotypes datasets
        pheno_data.index = self.int_genotypes.index
        pca_dataset = pd.concat([pheno_data,self.int_genotypes],axis=1)
        # Compute PCA
        pca = PCA(n_components=2, svd_solver='full')
        components = pca.fit_transform(pca_dataset)
        # Data viz
        f = plt.figure(figsize=(10,10))
        ax = f.add_subplot(111)
        color_dic = {"Pop1":"blue","Pop2":"red","Pop3":"yellow","Pop4":"green"}
        colors = [color_dic[i] for i in groups]
        ax.scatter([i[0] for i in components],[i[1] for i in components],c=colors)

exp = GWAS()
exp.generatePhenotypes()
exp.generateGenotypes()
exp.visualizePhenotypes()
exp.checkNormality()
exp.checkGenotypeSparsity()
exp.filterGenotypesSparsity()
exp.filterMAF()
exp.imputeMissing()
exp.analyzePopulationStructure()
