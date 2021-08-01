import numpy as np
import matplotlib.pyplot as plt
import pickle


"""
Plot performace metric plots of the training and predicting process

e.g: ELBO vs number of iterations, sl vs. number of iterations

"""



def plot_PerformanceMetrics():

    font = {"family":"serif",
        "size":12}
    plt.rc('font', **font)

    
    #Load File holding all the saved itteration data
    with open("Iteration_Info.pkl", "rb") as f:

        elbo_list = pickle.load(f)
        slx_list = pickle.load(f)
        sly_list =pickle.load(f)
        slz_list = pickle.load(f)
        scalefac_list = pickle.load(f)
        meanDens_list = pickle.load(f)


    #Plot ELBO vs. scale lengths   
    fig = plt.figure(figsize=(10, 8)) #width, height
    plt.plot(slx_list, elbo_list, color="black", linewidth=5, label="sl x")
    plt.plot(sly_list, elbo_list, color="blue", linewidth=5, label="sl y")
    plt.plot(slz_list, elbo_list, color="goldenrod", linewidth=5, label="sl z")
    plt.legend(fontsize=12, loc="upper right")
    plt.yscale("log")
    plt.xlabel("scale length [pc]")
    plt.ylabel("$-1 \\times $ ELBO")
    plt.savefig("ELBOvsSL.png")
    #plt.show()
    plt.close()

    #Plot ELBO vs. scale factor 
    fig = plt.figure(figsize=(10, 8)) #width, height
    plt.plot(scalefac_list, elbo_list, color="black", linewidth=5)
    plt.yscale("log")
    plt.xlabel("ln(scale factor)")
    plt.ylabel("$-1 \\times $ ELBO")
    plt.savefig("ELBOvsSFac.png")
    #plt.show()
    plt.close()


    #Plot ELBO vs. Mean Dense
    fig = plt.figure(figsize=(10, 8)) #width, height
    plt.plot(meanDens_list, elbo_list, color="black", linewidth=5)
    plt.yscale("log")
    plt.xlabel("mean density [log$_{10}$(mag/pc)]")
    plt.ylabel("$-1 \\times $ ELBO")
    plt.savefig("ELBOvsMeanDens.png")
    #plt.show()
    plt.close()


    #Iteration number (time evoultion) vs. ELBO
    fig = plt.figure(figsize=(10, 8)) #width, height
    plt.plot(elbo_list, color="black", linewidth=5)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("$-1 \\times $ ELBO")
    plt.savefig("Iter_vsELBO.png")
    #plt.show()
    plt.close()



    #Iteration number (time evoultion) vs. Scale Factor
    fig = plt.figure(figsize=(10, 8)) #width, height
    plt.plot(scalefac_list, color="black", linewidth=5)
    plt.xlabel("iteration")
    plt.ylabel("ln(scale factor)")
    plt.savefig("Iter_vsScaleFac.png")
    #plt.show()
    plt.close()


    #Iteration number (time evoultion) vs. meanDens
    fig = plt.figure(figsize=(10, 8)) #width, height
    plt.plot(meanDens_list, color="black", linewidth=5)
    plt.xlabel("iteration")
    plt.ylabel("mean density [log$_{10}$(mag/pc)]")
    plt.savefig("Iter_vsMeanDens.png")
    #plt.show()
    plt.close()

    #Iteration number (time evoultion) vs. SL:
    fig = plt.figure(figsize=(10, 8)) #width, height
    plt.plot(slx_list, color="black", linewidth=5, label="sl x")
    plt.plot(sly_list, color="blue", linewidth=5, label="sl y")
    plt.plot(slz_list, color="goldenrod", linewidth=5, label="sl z")
    plt.legend(fontsize=12, loc="upper right")
    plt.xlabel("iteration")
    plt.ylabel("scale length [pc]")
    plt.savefig("Iter_vsSL.png")
    #plt.show()
    plt.close()









