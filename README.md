# MixLIP

Matbench Discovery系のオープンソースMLIP（機械学習ポテンシャル）を統一インターフェースで扱い、知識蒸留や自前ファインチューニングができるツールキットです。

対応モデル: **MACE**, **CHGNet**, **SevenNet**, **EquiformerV2**, **M3GNet**, **ALIGNN**, **ORB**

---

## CPU環境（GPUなしノートPC）での開発セットアップ

GPU環境がなくても、軽量モデルとCPUモードで開発・テストをすべて行えます。

### 前提

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — PowerShellでのインストール:
  ```powershell
  pip install uv
  ```

### 1. 仮想環境を作成してアクティベート

```powershell
uv venv --python 3.11
.venv\Scripts\Activate.ps1
```

> [!NOTE]
> `Activate.ps1` が実行ポリシーエラーになる場合:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

### 2. PyTorchをCPU版でインストール

**GPU版と混在させないために、PyTorchは最初に単独でインストールします。**

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> [!NOTE]
> 将来GPU環境に切り替えるときは `--index-url https://download.pytorch.org/whl/cu121` に変えるだけです。

### 3. mixlipとコア依存関係をインストール

```powershell
# 開発モード + データ/訓練用パッケージ
uv pip install -e ".[data,train,dev]"

# torch_geometric (CPU版)
uv pip install torch_geometric
```

### 4. 使うバックエンドだけインストール

CPU上で動作確認しやすい軽量なモデルから始めることを推奨します。

```powershell
# --- 推奨: まずこの2つ ---
uv pip install -e ".[chgnet]"    # CHGNet: 最小依存、CPU速度が速い
uv pip install -e ".[matgl]"     # M3GNet: matgl経由、軽量

# --- 動作確認が取れたら ---
uv pip install -e ".[mace]"      # MACE: 高精度、CPUでもそこそこ動く
uv pip install -e ".[sevennet]"  # SevenNet
uv pip install -e ".[alignn]"    # ALIGNN
uv pip install -e ".[orb]"       # ORB
```

### 5. インストール確認

```powershell
python -c "import mixlip; print(mixlip.__version__)"
mixlip list-backends
```

---

## CPU環境での使い方

### 推論

すべての `load_calculator()` 呼び出しで `device="cpu"` を指定します。

```python
import mixlip

# CHGNet (CPU最速)
calc = mixlip.load_calculator("chgnet", device="cpu")

# MACE-MP small (CPU向け最小サイズ)
calc = mixlip.load_calculator("mace", device="cpu", model_size="small")

# M3GNet
calc = mixlip.load_calculator("m3gnet", device="cpu")
```

```python
from pymatgen.core import Structure, Lattice

# テスト用シリコン構造
si = Structure(Lattice.cubic(5.431), ["Si", "Si"], [[0,0,0],[0.25,0.25,0.25]])

result = calc.predict(si)
print(f"Energy: {result.energy:.4f} eV")
print(f"Forces:\n{result.forces}")
print(f"Stress: {result.stress}")
```

### 構造緩和 (CPU)

```python
from pymatgen.io.ase import AseAtomsAdaptor
from mixlip.inference import relax_structure

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

### データセット作成・ラベリング

```powershell
# Materials Projectから構造を取得
$env:MP_API_KEY = "your_api_key_here"
mixlip generate from-mp --formula "SrTiO3" --output data/srtio3.extxyz

# CHGNetでラベル付け (CPU)
mixlip generate label `
    --input data/srtio3.extxyz `
    --output data/srtio3_labeled.extxyz `
    --backend chgnet `
    --device cpu
```

### 評価

```powershell
mixlip eval `
    --backend chgnet `
    --dataset data/srtio3_labeled.extxyz `
    --device cpu `
    --n 50
```

---

## CPU環境での開発フロー

### テストを動かす

```powershell
# ユニットテスト (バックエンド不要、高速)
pytest tests/ -m "not integration" -v

# 特定モジュールのみ
pytest tests/test_training/test_loss.py -v
pytest tests/test_data/ -v
```

統合テスト（実際のモデルをロードするもの）は `@pytest.mark.integration` でマークされており、デフォルトでスキップされます。

```powershell
# 統合テストを含める場合 (CHGNetが必要)
pytest tests/ -m "integration" --backend chgnet -v
```

### 知識蒸留のドライラン

蒸留の設定が正しいか確認するために `fast_dev_run` を使います（GPUなしでも数秒で完了）。

```powershell
mixlip train --config configs/training/default.yaml --fast-dev-run
```

設定ファイルは事前に `device: cpu` に変更してください：

```yaml
# configs/training/default.yaml を編集
model:
  backend: chgnet
  device: cpu        # ← ここを変更
  ...
data:
  batch_size: 4      # ← CPU向けに小さくする
  num_workers: 0     # ← CPU環境ではWorker=0が安定
```

### Python APIで素早く試す

```python
from mixlip.data import AtomicSample, MLIPDataset
from mixlip.data.writers.extxyz_writer import write_extxyz
from mixlip.data.writers.hdf5_writer import write_hdf5, load_hdf5

# サンプルデータ作成
import numpy as np
from pymatgen.core import Structure, Lattice

si = Structure(Lattice.cubic(5.431), ["Si", "Si"], [[0,0,0],[0.25,0.25,0.25]])
samples = [
    AtomicSample(
        structure=si,
        energy=-21.69,
        forces=np.zeros((2, 3)),
        stress=np.array([0.01]*3 + [0.0]*3),
        source="test",
    )
    for _ in range(10)
]

# extxyz形式で保存
write_extxyz(samples, "test_data.extxyz")

# データセットとして読み込み (グラフ変換なし)
ds = MLIPDataset.from_extxyz("test_data.extxyz", as_graph=False)
print(f"{len(ds)} samples loaded")

# HDF5形式 (大規模データ向け)
write_hdf5(samples, "test_data.h5")
loaded = load_hdf5("test_data.h5")
```

---

## バックエンドのCPU速度目安

実際の速度はシステムや構造サイズによって変わります。

| モデル | CPU推論 (Si 2原子) | 依存の重さ |
|--------|-------------------|-----------|
| CHGNet | ~0.5s | 軽量 |
| M3GNet | ~0.3s | 軽量 |
| MACE-small | ~1–2s | 中程度 |
| MACE-medium | ~3–5s | 中程度 |
| SevenNet | ~1–3s | 中程度 |
| ALIGNN | ~2–5s | 中程度 |
| ORB | ~1–2s | 中程度 |

> CPU開発では CHGNet か M3GNet でロジックを確認し、GPU環境で MACE-medium/large に切り替えるワークフローが効率的です。

---

## ディレクトリ構成

```
mixlip/
├── pyproject.toml
├── configs/
│   ├── models/          # モデルごとの設定テンプレート
│   └── training/        # 訓練・蒸留設定テンプレート
├── src/mixlip/
│   ├── core/            # Protocol, Config, Registry
│   ├── calculators/     # バックエンドアダプタ (1ファイル/モデル)
│   ├── data/            # AtomicSample, Dataset, Loaders, Writers
│   ├── training/        # Loss, LightningModule, Distillation
│   ├── inference/       # Ensemble, Relax, MD
│   └── cli/             # CLIコマンド
└── tests/
```

## トラブルシューティング

**`torch_geometric` がインポートエラーになる場合**

`torch_geometric` のバージョンはインストールしたPyTorchのバージョンと合わせる必要があります。

```powershell
python -c "import torch; print(torch.__version__)"
# 例: 2.3.0+cpu

# バージョンを合わせてインストール
uv pip install torch_geometric
# torch-scatter/sparse が必要な場合
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```

**CHGNetがCUDA関連のエラーを出す場合**

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 先頭に追加

calc = mixlip.load_calculator("chgnet", device="cpu")
```

**MACEが `default_dtype` エラーを出す場合**

```python
calc = mixlip.load_calculator(
    "mace", device="cpu", model_size="small", dtype="float64"
)
```
