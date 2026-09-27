# MixLIP はじめてのチュートリアル

MixLIP を初めて触る人向けの、手を動かしながら学ぶチュートリアルです。GPU なしの CPU 環境を前提にしています。より詳しいリファレンス(全バックエンドの一覧、トラブルシューティングなど)は [README.md](README.md) を参照してください。

このチュートリアルで扱う流れ:

1. セットアップ
2. 最初の推論
3. 構造緩和
4. 自分のデータセットを作って精度を測る
5. ミニ・ファインチューニング
6. 知識蒸留を体験する

各ステップのコードは、実際にこのリポジトリの CPU 環境で動作確認済みです。

---

## 1. セットアップ

### 前提

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

```powershell
pip install uv
```

### 仮想環境を作る

```powershell
uv venv --python 3.11
.venv\Scripts\Activate.ps1
```

### PyTorch(CPU版)→ mixlip → CHGNet の順にインストール

このチュートリアルでは **CHGNet** だけを使います。理由は 2 つあります。

- CPU でも高速に動く軽量なモデルであること
- `mixlip train`/`mixlip distill run`(ファインチューニング・知識蒸留)が現状 **CHGNet のみ対応**であること(他のバックエンドは推論・蒸留の教師役としては使えますが、学習対象にはまだできません)

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[data,train,dev]"
uv pip install torch_geometric
uv pip install -e ".[chgnet]"

# 知識蒸留のステップで教師モデルとして使うので、M3GNetも入れておきます
uv pip install -e ".[matgl]"
```

### インストール確認

```powershell
mixlip list-backends
```

`chgnet` を含む7つのバックエンド名が表示されれば成功です(実際に使えるのは、対応パッケージをインストールしたものだけです)。

---

## 2. 最初の推論

シリコンの結晶構造を作り、CHGNet でエネルギー・力・応力を予測します。

```python
import mixlip
from pymatgen.core import Structure, Lattice

si = Structure(Lattice.cubic(5.431), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
calc = mixlip.load_calculator("chgnet", device="cpu")
result = calc.predict(si)

print(f"Energy: {result.energy:.4f} eV")
print(f"Forces shape: {result.forces.shape}")
print(f"Stress shape: {result.stress.shape}")
```

実行結果の例:

```
Energy: -6.2461 eV
Forces shape: (2, 3)
Stress shape: (6,)
```

> [!NOTE]
> このサンプルの Si 構造(立方格子に原子2個)は動作確認用の非物理的な配置で、実際のダイヤモンド構造ではありません。次のステップの構造緩和で、より安定な形に緩和されます。

---

## 3. 構造緩和

`relax_structure` を使うと、力が十分小さくなるまで原子位置とセルを最適化できます。

```python
import mixlip
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from mixlip.inference import relax_structure

si = Structure(Lattice.cubic(5.431), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
calc = mixlip.load_calculator("chgnet", device="cpu")

atoms = AseAtomsAdaptor.get_atoms(si)
relaxed, info = relax_structure(
    atoms,
    calculator=calc,
    fmax=0.05,
    max_steps=100,
    relax_cell=True,
)
print(info)
```

実行結果の例:

```
{'converged': True, 'n_steps': 50, 'final_energy_eV': -10.6272, 'final_fmax_eV_per_A': 0.0069}
```

`converged: True` となり、格子定数が緩和されてエネルギーが下がっていることを確認してください。

---

## 4. 自分のデータセットを作って精度を測る

`AtomicSample` にラベル(energy/forces/stress)を持たせて `extxyz` 形式で保存し、`mixlip eval` でモデルの精度を確認します。

```python
import numpy as np
from pymatgen.core import Structure, Lattice
from mixlip.data.schema import AtomicSample
from mixlip.data.writers.extxyz_writer import write_extxyz

si = Structure(Lattice.cubic(5.431), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

# ここでは本物のDFT計算の代わりに、仮のラベルを5サンプル分入れています。
samples = [
    AtomicSample(
        structure=si,
        energy=-21.69,
        forces=np.zeros((2, 3)),
        stress=np.array([0.01, 0.01, 0.01, 0.0, 0.0, 0.0]),
        source="tutorial",
    )
    for _ in range(5)
]
write_extxyz(samples, "data/my_dataset.extxyz")
```

```powershell
mixlip eval --backend chgnet --dataset data/my_dataset.extxyz --device cpu --n 5
```

```
      chgnet — my_dataset.extxyz
┌─────────────────────────┬───────────┐
│ Metric                  │     Value │
├─────────────────────────┼───────────┤
│ energy_mae_meV_per_atom │ 7721.9703 │
│ forces_mae_meV_per_A    │  250.8497 │
│ forces_rmse_meV_per_A   │  250.8497 │
│ forces_cos_sim          │    0.0000 │
│ stress_mae_GPa          │    0.3631 │
└─────────────────────────┘
```

> [!NOTE]
> ここでの誤差が大きいのは、`-21.69` という仮のラベル値が実際の DFT 計算値ではなく、ワークフローを確認するためのダミーだからです。本物のデータセットで評価すればこの値は意味を持ちます。VASP や Materials Project のデータを使う方法は README の「データセット作成・ラベリング」を参照してください。

---

## 5. ミニ・ファインチューニング

学習を試すために、`--fast-dev-run`(1バッチだけ実行して即終了するデバッグモード)で CHGNet をファインチューニングしてみます。まず `configs/training/default.yaml` を手元用にコピーして、CPU用に調整します。

`my_first_finetune.yaml`:

```yaml
model:
  backend: chgnet
  checkpoint: pretrained
  device: cpu          # ← GPUがない場合はcpu
  dtype: float32
  compute_stress: true

data:
  train_path: data/my_dataset.extxyz
  val_path: data/my_dataset.extxyz   # 本来はtrainと別のファイルにします
  format: extxyz
  batch_size: 2         # ← CPU向けに小さく
  num_workers: 0         # ← CPU環境ではWorker=0が安定

loss:
  energy_weight: 1.0
  forces_weight: 100.0
  stress_weight: 10.0
  criterion: huber
  huber_delta: 0.01

optimizer: adamw
lr: 1.0e-4
max_epochs: 1
use_ema: false
scheduler: cosine
```

```powershell
mixlip train --config my_first_finetune.yaml --fast-dev-run
```

```
        Epoch 0
┌────────────┬────────┐
│ Metric     │  Value │
├────────────┼────────┤
│ val/energy │ 0.0771 │
│ val/forces │ 0.2307 │
│ val/stress │ 0.0000 │
│ val/total  │ 0.3079 │
└────────────┴────────┘
Training complete.
```

`--fast-dev-run` を外し `max_epochs` を増やせば、本格的な学習になります。CHGNet 以外のバックエンド(`mace` など)を `backend:` に指定すると、この時点でトレースバックではなく「Training is not yet supported for backend '...'. Currently supported: chgnet.」という明快なエラーで止まります — これは仕様です。

---

## 6. 知識蒸留を体験する

知識蒸留は、大きい/高精度なモデル(**教師**)の予測を、小さい/軽量なモデル(**生徒**)に学習させる仕組みです。教師は推論(`.predict()`)しか使わないので7バックエンドどれでもなれますが、生徒は実際に学習する必要があるため、現状は CHGNet のみです。ここでは M3GNet を教師、CHGNet を生徒にします。

### 6-1. 教師ラベルを事前計算する

毎回教師モデルを呼び出す代わりに、先にラベルを計算して保存しておく(オフライン蒸留)方法です。速く、教師モデルをメモリに持たなくて済みます。

```powershell
mixlip distill generate-labels --teacher m3gnet --input data/my_dataset.extxyz --output data/labeled.h5 --device cpu
```

### 6-2. 蒸留学習を実行する

`my_first_distill.yaml`:

```yaml
model:
  backend: chgnet   # distill.studentと同じにしておく(このフィールド自体は蒸留では直接使われません)
  checkpoint: pretrained
  device: cpu
  dtype: float32
  compute_stress: true

data:
  train_path: data/labeled.h5
  val_path: data/labeled.h5
  format: hdf5
  batch_size: 2
  num_workers: 0

loss:
  energy_weight: 1.0
  forces_weight: 100.0
  stress_weight: 10.0
  criterion: huber
  huber_delta: 0.01

optimizer: adamw
lr: 1.0e-4
max_epochs: 1
use_ema: false
scheduler: cosine

distill:
  teacher:
    backend: m3gnet
    checkpoint: pretrained
    device: cpu
    dtype: float32

  student:
    backend: chgnet
    checkpoint: pretrained
    device: cpu
    dtype: float32

  alpha_task: 0.5      # DFTラベルとの誤差の重み
  alpha_distill: 0.5   # 教師予測との誤差の重み

  loss:
    energy_weight: 1.0
    forces_weight: 100.0
    stress_weight: 10.0
    criterion: huber
    huber_delta: 0.01
```

```powershell
mixlip distill run --config my_first_distill.yaml
```

```
Distillation:
  Teacher: m3gnet
  Student: chgnet
...
Distillation complete.
```

`data/labeled.h5` に教師ラベルが入っているので、この学習中に M3GNet が推論のために呼ばれることはありません(`sample.metadata` に保存された `teacher_energy`/`teacher_forces` がそのまま使われます)。

---

## 次のステップ

- 全バックエンドの一覧、CPU速度の目安、トラブルシューティングは [README.md](README.md) を参照してください。
- 本物の DFT データを使う場合は、README の「データセット作成・ラベリング」(Materials Project からの取得や VASP 出力の読み込み)を参照してください。
- CHGNet 以外のバックエンドをファインチューニング対応させたい場合は、`src/mixlip/calculators/chgnet.py` の `trainable_module`/`training_forward` の実装がそのままパターンになります。
