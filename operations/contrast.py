import numpy as np
import matplotlib.pyplot as plt

def contrast(img, fac=1., theta=1., phi=1., maxIntensity=255.):
    x = np.arange(maxIntensity) 

    res = np.zeros_like(img)
    for i in [0,1,2]:
        res[:,:,i] = (maxIntensity/phi)*(img[:,:,i]/(maxIntensity/theta))**fac

    fig = plt.figure()
    plt.plot(np.arange(0.,255), (maxIntensity/phi)*(np.arange(0.,255.)/(maxIntensity/theta))**fac, '.')
    plt.plot([0., 255.], [0., 255.], 'k--')
    plt.title("Contrast value: %.1f" % fac)

    return res
