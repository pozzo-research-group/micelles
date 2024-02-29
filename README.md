## Form factors of block copolymer micelles with spherical, ellipsoidal and cylindrical cores

This repository contains Python code to compute form factors of micelles described in the following paper:

```bib
@article{pedersen2000form,
  title={Form factors of block copolymer micelles with spherical, ellipsoidal and cylindrical cores},
  author={Pedersen, Jan Skov},
  journal={Journal of Applied Crystallography},
  volume={33},
  number={3},
  pages={637--640},
  year={2000},
  publisher={International Union of Crystallography}
}
```

## Using these modes in a GUI (MacOS)
You can use these custom models in SasView GUI using their `plugin_models` capabilities. Follow the instructions below to make the plugin models available for use in a GUI.

1. Locate the following folder on your Mac hard drive: `/users/<your_username>/.sasview/plugin_models/`. 
Note that to locate the`.` folders, you need to press `Command + Shift + . (the period key)` after you are in the `/users/` folder.

2. You can now copy the Python files related to the models into this folder and they should be available for use when you reopen SasView. 

## Using models in sasmodels (Python)

Install the required packages using `pip install -r requirements.txt`

```python
import numpy as np
from sasmodels.core import load_model
from sasmodels.direct_model import call_kernel

model = load_model("../models/spherical_micelle.py")
q = np.logspace(-3,0, num=1000)
kernel = model.make_kernel([q])
Iq_sasmodels = call_kernel(kernel, sphere_params) # provide parameters approporiate to the model
```

### Notes:
Although sasmodels require everything to be in Angstroms and the resulting intensity is in cm^-1, these plugin models do not require nor perform any unit conversions. The intensities output by the `call_kernel` method above are in inverse length scale.
