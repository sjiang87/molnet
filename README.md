# molnet
### Author: Shengli Jiang (sjiang87@wisc.edu)

A deephyer neural architecture search problem for molecule-net benchmark.
- load_data.py
- search_space.py
- problem.py

```bash
srun -n 30 python -m tuster.system.bebop.run 'python -m deephyper.search.nas.regevo --evaluator ray --redis-address {redis_address} --problem molnet.molnet.problem.Problem'
```
