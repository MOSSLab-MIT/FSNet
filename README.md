## FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees
This repository is by 
[Hoang T. Nguyen](https://www.linkedin.com/in/hoang-nguyen-971519201/) and 
[Priya L. Donti](https://www.priyadonti.com)
 and contains source code to reproduce the experiments in our paper 
 ["FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees"](https://arxiv.org/abs/2506.00362).


## Abstract
<p style="text-align: justify;">
Efficiently solving constrained optimization problems is crucial for numerous real-world applications, yet traditional solvers are often computationally prohibitive for real-time use. Machine learning-based approaches have emerged as a promising alternative to provide approximate solutions at faster speeds, but they struggle to strictly enforce constraints, leading to infeasible solutions in practice. To address this, we propose the Feasibility-Seeking-Integrated Neural Network (FSNet), which integrates a feasibility-seeking step directly into its solution procedure to ensure constraint satisfaction. This feasibility-seeking step solves an unconstrained optimization problem that minimizes constraint violations in a differentiable manner, enabling end-to-end training and providing guarantees on feasibility and convergence. Our experiments across a range of different optimization problems, including both smooth/nonsmooth and convex/nonconvex problems, demonstrate that FSNet can provide feasible solutions with solution quality comparable to (or in some cases better than) traditional solvers, at significantly faster speeds. 

<p align="center">
  <img src="figures\diagram.png" alt="FSNet Diagram" width="800"/>
</p>


If you find this repository helpful in your publications, please consider citing our paper.
```bash
@article{nguyen2025fsnet,
    title={FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees}, 
    author={Hoang T. Nguyen and Priya L. Donti},
    year={2025},
    journal={arXiv preprint arXiv:2506.00362},
}
```


## 🚀 Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎓 Usage

### Training and Test

```bash
python main.py \
  --method <FSNet|penalty|adaptive_penalty|DC3|projection> \
  --prob_type <convex|nonconvex|nonsmooth_nonconvex> \
  --prob_name <qp|qcqp|socp>
```

* `--method`

  * `FSNet`              (Feasibility-Seeking Neural Network)
  * `penalty`            (Penalty method)
  * `adaptive_penalty`   (Adaptive Penalty method)
  * `DC3`                (Differentiable Convex–Concave Correction, DC3)
  * `projection`         (Projection-based method; supported for QP only)
* `--prob_type`

  * `convex`
  * `nonconvex`
  * `nonsmooth_nonconvex`
* `--prob_name`

  * `qp`   (Quadratic Program)
  * `qcqp` (Quadratically Constrained Quadratic Program)
  * `socp` (Second-Order Cone Program)
* And see `main.py` for more relevant flags.

Example:
```bash
python main.py --method FSNet --prob_type convex --prob_name qp
```
