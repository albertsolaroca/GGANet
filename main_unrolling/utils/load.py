# # Libraries
# import pickle
# import wntr
# from tkinter import Tk #Window explorer for the file
# from tkinter.filedialog import askopenfilename
# import os


# def load_wdn(wdn_names=None, size=None, show=True):
    # '''
    # Loads one or more water distribution network datasets
    # if no list is given, the user can select one network
    # -------
    # wdn_names : list
        # list of water distribution network names
    # '''
    # if wdn_names is None:
        # # Load single network
        # # Open a water network .inp file
        # Tk().withdraw() 
        # inp_file = askopenfilename(title='Select a network', filetypes=[('inp files', '*.inp')])

        # wdn_name = inp_file[-7:-4]
        # database_file = inp_file[:-4]+'_'+str(size)+'.p'

        # # Import database: contains information for all the simulations run
        # database = pickle.load( open(database_file, "rb" ))

        # wn_WDS = wntr.network.WaterNetworkModel(inp_file)

        # if show:
            # # Graph the network
            # title = wdn_name+"\nNumber of nodes:{} Number of edges:{}".format(wn_WDS.num_nodes, wn_WDS.num_links)
            # wntr.graphics.plot_network(wn_WDS, title=title);

        # wdns = {wdn_name : wn_WDS.num_nodes}
            
        # return database, wdns
    
    # else:
        # # Load multiple networks
        # curr_dir = os.getcwd()
        # databases = []
        # wdns = {}

        # for wdn_name in wdn_names:
            # inp_file = '{}/Database/Datasets/{}/{}.inp'.format(curr_dir,wdn_name,wdn_name)
            # # info_file = inp_file[:-4]+'_info.csv'
            # database_file = inp_file[:-4]+'_'+str(size)+'.p'

            # # Import database: contains information for all the simulations run
            # database_i = pickle.load( open(database_file, "rb" ))

            # databases += database_i

            # wn_WDS = wntr.network.WaterNetworkModel(inp_file)

            # if show:
                # # Graph the network
                # title = wdn_name+"\nNumber of nodes:{} Number of edges:{}".format(wn_WDS.num_nodes, wn_WDS.num_links)
                # wntr.graphics.plot_network(wn_WDS, title=title);

            # wdns[wdn_name] = wn_WDS.num_nodes
            
        # return databases, wdns
	
	
# def load_new_dataset(database_file=None):
    # '''
    # Loads database of simulations
    # If no name is give, the user can choose a database .p file. In this case, the name is returned as additional output
    # -------
    # database_file: str
        # if given, it indicates a database in pickle format
        # the name is composed of the networks' names + number of simulations, seprated by _
        # e.g., name = BLA_BAK_FOS_PES_MOD_BIN_100
    # '''
    # if database_file is None:
        # # Load and import database
        # Tk().withdraw() 
        # database_file = askopenfilename(title='Select a database', filetypes=[('pickle file', '*.p')])
        # database = pickle.load(open(database_file, "rb" ))
        
        # return database, database_file.split('/')[-1]
    
    # else:
        # curdir = os.getcwd() + '\\database\\datasets\\'
        # database = pickle.load(open(f'{curdir}{database_file}.p', "rb" ))
            
        # return database
	
	
# def get_wdns_info(database, database_file):
    # '''
    # This function provides a dictionary of the networks given the database and its file name
    # -----
    # database: list
        # list of network simulations
    # database_file: str
        # name of the database e.g., 'BLA_BAK_FOS_PES_MOD_BIN_100'
    # '''
    # wdns = {}
    
    # for dataset in database:
        # for name in database_file.split('_')[:-1]:
            # if dataset.name == name:
                # wdns[name] = dataset.elevation.shape[0]
                # continue
                
    # return wdns


# def extract_database(database, wdn_name=None, wdns=None):
    # '''
    # Extract single WDN from database of simulations
    # ------
    # database: list
        # list of Data objects, containing wdn simulations
    # wdn_name: str
        # name of a WDN in database
	# wdns: dict
		# dictionary of wdns names
    # '''
    # database_red = []
    # for data in database:
        # if data.name == wdn_name:
            # database_red.append(data)
    
    # if wdns is not None:
        # new_wdns = {wdn_name: wdns[wdn_name]}
        # return database_red, new_wdns
    
    # else:
        # return database_red
        

def load_wdn(wdn_name, data_folder):
    '''
    Load one owater distribution network datasets
    if no list is given, the user can select one network
    -------
    wdn_names : list
        list of water distribution network names
    '''
    if wdn_names is None:
        # Load single network
        # Open a water network .inp file
        Tk().withdraw() 
        inp_file = askopenfilename(title='Select a network', filetypes=[('inp files', '*.inp')])

        wdn_name = inp_file[-7:-4]
        database_file = inp_file[:-4]+'_'+str(size)+'.p'

        # Import database: contains information for all the simulations run
        database = pickle.load( open(database_file, "rb" ))

        wn_WDS = wntr.network.WaterNetworkModel(inp_file)

        if show:
            # Graph the network
            title = wdn_name+"\nNumber of nodes:{} Number of edges:{}".format(wn_WDS.num_nodes, wn_WDS.num_links)
            wntr.graphics.plot_network(wn_WDS, title=title);

        wdns = {wdn_name : wn_WDS.num_nodes}
            
        return database, wdns
    
    else:
        # Load multiple networks
        curr_dir = os.getcwd()
        databases = []
        wdns = {}

        for wdn_name in wdn_names:
            inp_file = '{}/Database/Datasets/{}/{}.inp'.format(curr_dir,wdn_name,wdn_name)
            # info_file = inp_file[:-4]+'_info.csv'
            database_file = inp_file[:-4]+'_'+str(size)+'.p'

            # Import database: contains information for all the simulations run
            database_i = pickle.load( open(database_file, "rb" ))

            databases += database_i

            wn_WDS = wntr.network.WaterNetworkModel(inp_file)

            if show:
                # Graph the network
                title = wdn_name+"\nNumber of nodes:{} Number of edges:{}".format(wn_WDS.num_nodes, wn_WDS.num_links)
                wntr.graphics.plot_network(wn_WDS, title=title);

            wdns[wdn_name] = wn_WDS.num_nodes
            
        return databases, wdns
        