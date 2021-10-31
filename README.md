<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/radioactive11/amethyst">
    <img src="IMAGES/amethyst.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">amethyst</h3>

  <p align="center">
    A low-code recommendation engine generation tool
    <br />
    <a href="https://github.com/radioactive11/amethyst"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/radioactive11/amethyst">View Demo</a>
    ·
    <a href="https://github.com/radioactive11/amethyst/issues">Report Bug</a>
    ·
    <a href="https://github.com/radioactive11/amethyst/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<img src="IMAGES/amethyst-flowchart.png" alt="Desc.">

There are a few ways to deal with the challenge of designing recommendation engines. One is to have your own team of engineers and
data scientists, all highly trained in machine learning, to custom design recommenders to meet your needs.

Smaller companies, typically startups, look for an easier solution to design a recommendation engine wihtout spending much resources.

Not only that, many developers, who do not have a machine learning/data science background want to integrate recommenders in their product.


Amethyst aims to solve this problem by providing a low-code recommendation engine generator which is not only easy to train but also efforless to deploy.

It requires only three parameters to rank/predict best items for users and vice-versa
* User ID (unique identifier for each user)
* Item ID (unique identifier for each item)
* User-Item Ratings (user-item rating/interaction scores)



<p align="right">(<a href="#top">back to top</a>)</p>



### Built With
<p align="left">
<img alt="pytorch" src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"/>
<img alt="pytorch" src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img alt="pytorch" src="https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
</p>

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you can generate your own collaborative recommendation engine.
To get a local copy up and running follow these simple example steps.

### Prerequisites

* Python>=3.7


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/radioactive11/amethyst.git
   ```
2. Create and activate virtual environment
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the tool
   ```sh
   python3 setup.py install
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

A recommendation engine can be generated in 4 easy steps:
1. Import the data
2. Select an algorithm
3. Train the model
4. Evaluate the model's performmance

### Data Split ⚗️
```py
from amethyst.dataloader import split

df = pd.read_csv("./movielens100k.csv")
df_train, df_test = split.stratified_split(
                                    df,
                                    0.8, 
                                    user_col='userID',
                                    item_col='itemID',
                                    filter_col='item'
)
```

### Load Data 📥

```py
from amethyst.dataloader import dataset

df = pd.read_csv("movielens100k.csv")

# from Data Split
df_train, df_test = split.stratified_split(df)

train = dataset.Dataloader.dataloader(df_train.itertuples(index=False))
test = dataset.Dataloader.dataloader(df_test.itertuples(index=False))

```

### Train (BiVAECF) ⚙️

```py

from amethyst.models.bivaecf.bivaecf import BiVAECF
import torch


bivae = BiVAECF(
    k=50,
    encoder_structure=[100],
    act_fn=["tanh"],
    likelihood="pois",
    n_epochs=500,
    batch_size=256,
    learning_rate=0.001,
    seed=42,
    use_gpu=torch.cuda.is_available(),
    verbose=True
)

bivae.fit(train, test)
bivae.save("model.pkl")
```

### Train (IBPR) ⚙️

```py

from amethyst.models.ibpr.ibprcf import IBPR
import torch


ibpr = IBPR(
        k=20,
        max_iter=100,
        alpha_=0.05,
        lambda_=0.001,
        batch_size=100,
        trainable=True,
        verbose=False,
        init_params=None)

ibpr.fit(train, test)
ibpr.save("model.pkl")
```

### Predict/Rank 📈

```py

from amethyst.models.predictions import rank
from amethyst.models.bivaecf.bivaecf import BiVAECF


bivae = BiVAECF(
    k=50,
    encoder_structure=[100],
    act_fn=["tanh"],
    likelihood="pois",
    n_epochs=500,
    batch_size=256,
    learning_rate=0.001,
    seed=42,
    use_gpu=torch.cuda.is_available(),
    verbose=True
)

bivae.load("mode.pkl")

predictions = rank(bivae, test, user_col='userID', item_col='itemID')

# predictions is a Pandas Dataframe
predictions.to_csv("predictions.csv", index=False)
```


### Evaluate 📈

```py

from amethyst.models.predictions import rank
from amethyst.eval.eval_methods import map_at_k, precision_at_k, recall_k



bivae = BiVAECF(
    k=50,
    encoder_structure=[100],
    act_fn=["tanh"],
    likelihood="pois",
    n_epochs=500,
    batch_size=256,
    learning_rate=0.001,
    seed=42,
    use_gpu=torch.cuda.is_available(),
    verbose=True
)

bivae.load("mode.pkl")

predictions = rank(bivae, test, user_col='userID', item_col='itemID')
eval_map = map_at_k(test, predictions, k=10)
pk = precision_at_k(test, predictions, k=10)
rk = recall_k(test, predictions)

```


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [] Add Additional Templates w/ Examples
- [] Add "components" document to easily copy & paste sections of the readme
- [] Multi-language Support
    - [] Chinese
    - [] Spanish

See the [open issues](https://github.com/radioactive11/amethyst/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Arijit Roy - [@your_twitter](https://twitter.com/your_username) - roy.arijit2001@gmail.com

Project Link: [https://github.com/radioactive11/amethyst](https://github.com/radioactive11/amethyst)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/radioactive11/amethyst.svg?style=for-the-badge
[contributors-url]: https://github.com/radioactive11/amethyst/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/radioactive11/amethyst.svg?style=for-the-badge
[forks-url]: https://github.com/radioactive11/amethyst/network/members
[stars-shield]: https://img.shields.io/github/stars/radioactive11/amethyst.svg?style=for-the-badge
[stars-url]: https://github.com/radioactive11/amethyst/stargazers
[issues-shield]: https://img.shields.io/github/issues/radioactive11/amethyst.svg?style=for-the-badge
[issues-url]: https://github.com/radioactive11/amethyst/issues
[license-shield]: https://img.shields.io/github/license/radioactive11/amethyst.svg?style=for-the-badge
[license-url]: https://github.com/radioactive11/amethyst/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/arijit--roy
[product-screenshot]: images/screenshot.png