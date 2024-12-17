import argparse
import json
import os
import pickle
import re
import shutil
import signal
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from be_great import GReaT
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
from scipy import stats
from scipy.stats import wasserstein_distance
from sdmetrics.column_pairs import CorrelationSimilarity
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_column import KSComplement
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import Metadata, SingleTableMetadata
from sdv.single_table import (
    CopulaGANSynthesizer,
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)
from sklearn.model_selection import train_test_split
from snsynth import Synthesizer

import wandb

console = Console()


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out")


class SyntheticDataPipeline:
    def __init__(
        self,
        data_path: str,
        output_dir: str,
        experiment_name: str,
        use_wandb: bool = False,
        small_training: bool = False,
        selected_models: List[str] = None,
        epochs: int = 300,
        epochs_great: int = 10,
        time_limit: int = 3600,
        verbose: bool = False,
    ):
        self.data_path = data_path
        self.output_dir = os.path.join(output_dir, experiment_name)
        self.experiment_name = experiment_name
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.metadata = None
        self.models = {}
        self.results = {}
        self.use_wandb = use_wandb
        self.small_training = small_training
        self.epochs = epochs
        self.epochs_great = epochs_great
        self.selected_models = selected_models or [
            "PATE-CTGAN",
            "DP-CTGAN",
            # "MWEM",
            "GaussianCopula",
            "CTGAN",
            "TVAE",
            "CopulaGAN",
            "GReaT",
        ]
        self.time_limit = time_limit
        self.verbose = verbose

        if self.small_training:
            self.epochs = 2
        # Create output directories
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)  # Remove existing output directory
        os.makedirs(self.output_dir, exist_ok=True)
        for model in self.selected_models:
            os.makedirs(os.path.join(self.output_dir, model), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "comparison"), exist_ok=True)

    def load_data(self):
        console.print("[bold blue]Loading data...")

        # Detect the delimiter
        with open(self.data_path, "r") as file:
            first_line = file.readline()
            if "," in first_line:
                delimiter = ","
            elif ";" in first_line:
                delimiter = ";"
            else:
                raise ValueError(
                    "Unable to detect delimiter. File should use either ',' or ';' as delimiter."
                )

        # Read the CSV file
        self.data = pd.read_csv(self.data_path, sep=delimiter)

        # Drop the 'Unnamed: 0' column if it exists
        if "Unnamed: 0" in self.data.columns:
            self.data = self.data.drop("Unnamed: 0", axis=1)
            console.print(
                "[yellow]Dropped 'Unnamed: 0' column (likely an unnecessary index)."
            )

        # drop columns with missing values
        self.data = self.data.dropna()

        # drop columns first_name, last_name, street_name, locality_name, house_number
        columns_to_drop = [
            "first_name",
            "last_name",
            "street_name",
            "locality_name",
            "house_number",
        ]
        self.data = self.data.drop(
            columns=[col for col in columns_to_drop if col in self.data.columns]
        )

        initial_rows = len(self.data)

        console.print(f"Initial number of rows: {initial_rows}")
        console.print(f"Number of columns: {len(self.data.columns)}")
        console.print(f"Columns: {', '.join(self.data.columns)}")

        # Handle small training if needed
        if self.small_training:
            n_samples = min(5000, len(self.data))
            self.data = self.data.sample(n=n_samples, random_state=42)
            console.print(f"Using {n_samples} samples for small training")

        # Split data into train, validation, and test sets
        train_val, self.test_data = train_test_split(
            self.data, test_size=0.2, random_state=42
        )
        self.train_data, self.val_data = train_test_split(
            train_val, test_size=0.25, random_state=42
        )  # 0.25 x 0.8 = 0.2

        # Create metadata
        self.metadata = SingleTableMetadata()
        for column in self.data.columns:
            if (
                self.data[column].dtype == "object"
                or self.data[column].dtype.name == "category"
            ):
                self.metadata.add_column(column, sdtype="categorical")
            elif np.issubdtype(self.data[column].dtype, np.number):
                self.metadata.add_column(column, sdtype="numerical")
            else:
                self.metadata.add_column(column, sdtype="unknown")

        # Save metadata
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        self.metadata.save_to_json(metadata_path)

        console.print("[bold green]Data loaded successfully!")

    def preprocess_data(self):
        console.print("[bold blue]Starting data preprocessing...")

        initial_rows = len(self.data)
        console.print(f"Initial number of rows: {initial_rows}")

        # Drop unnecessary columns
        columns_to_drop = [
            "first_name",
            "last_name",
            "street_name",
            "locality_name",
            "house_number",
        ]
        self.data = self.data.drop(columns=columns_to_drop)
        console.print(f"Columns dropped: {', '.join(columns_to_drop)}")

        # Convert numeric columns to appropriate types
        numeric_columns = [
            "salary",
            "rent",
            "age",
            "north_coord",
            "east_coord",
            "postal_code",
            "height",
            "weight",
            "z1",
            "z2",
        ]
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        # Handle categorical columns
        categorical_columns = ["gender", "canton", "building_category"]
        for col in categorical_columns:
            self.data[col] = self.data[col].astype("category")

        # Drop rows with NaN values
        self.data = self.data.dropna()

        final_rows = len(self.data)
        rows_dropped = initial_rows - final_rows
        console.print(f"Rows dropped: {rows_dropped}")
        console.print(f"Final number of rows: {final_rows}")

        if final_rows == 0:
            console.print(
                "[bold red]Warning: All rows were dropped during preprocessing!"
            )
            console.print("Please check your data and preprocessing steps.")
            return

        console.print("[bold green]Data preprocessed successfully!")

    def prepare_models(self):
        console.print("[bold green]Preparing models...")
        all_models = {}
        if not self.small_training:
            all_models["GReaT"] = lambda: GReaT(
                llm="distilgpt2",
                batch_size=32,
                epochs=self.epochs_great,
                fp16=False,
                efficient_finetuning="lora",
            )
        all_models["GaussianCopula"] = lambda: GaussianCopulaSynthesizer(
            metadata=self.metadata
        )
        all_models["CTGAN"] = lambda: CTGANSynthesizer(
            metadata=self.metadata, epochs=self.epochs, verbose=self.verbose
        )
        all_models["TVAE"] = lambda: TVAESynthesizer(
            metadata=self.metadata, epochs=self.epochs, verbose=self.verbose
        )
        all_models["CopulaGAN"] = lambda: CopulaGANSynthesizer(
            metadata=self.metadata, epochs=self.epochs, verbose=self.verbose
        )
        all_models["PATE-CTGAN"] = lambda: Synthesizer.create(
            "patectgan", epsilon=1.0, delta=1e-5, verbose=self.verbose
        )

        all_models["DP-CTGAN"] = lambda: Synthesizer.create(
            "dpctgan", epsilon=1.0, delta=1e-5, sigma=2, verbose=self.verbose
        )

        all_models["MWEM"] = lambda: Synthesizer.create(
            "mwem", epsilon=1.0, verbose=self.verbose
        )

        self.models = {
            name: all_models[name]()
            for name in self.selected_models
            if name in all_models
        }

        console.print("[bold green]Models prepared successfully!")

    def dry_validation_run(self):
        console.print("[bold blue]Performing dry validation run...")
        dry_run_data = {"Original": self.val_data}
        results = self.evaluate_models(dry_run_data)
        self.display_results(results, "Score evaluation test vs validation data")

    def train_models(self):
        for name, model in self.models.items():
            console.print(f"[green]Training {name}...")
            start_time = time.time()

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.time_limit)  # 1 hour timeout

            try:
                if name in ["PATE-CTGAN", "DP-CTGAN", "MWEM"]:
                    # DP models need features and labels separately
                    model.fit(
                        self.train_data,
                        self.train_data.columns,
                        preprocessor_eps=0.5,
                    )
                else:
                    model.fit(self.train_data)

            except TimeoutException:
                console.print(f"[yellow]Training for {name} timed out after 1 hour.")
            finally:
                signal.alarm(0)

            end_time = time.time()
            training_time = end_time - start_time
            console.print(
                f"[bold green]Training time for {name}: {training_time:.2f} seconds"
            )
            if self.use_wandb:
                wandb.log({f"{name}_training_time": training_time})

    def generate_synthetic_data(self, gen_num_samples: int) -> Dict[str, pd.DataFrame]:
        synthetic_data = {}
        for name, model in self.models.items():
            console.print(f"[green]Generating data with {name}...")
            start_time = time.time()

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.time_limit)

            try:
                if isinstance(model, GReaT):
                    if self.small_training:
                        synthetic_data[name] = (
                            self.train_data
                        )  # Use training data for small training
                        console.print(
                            "[yellow]Using training data for GReaT in small training mode."
                        )
                    else:
                        synthetic_data[name] = model.sample(
                            n_samples=gen_num_samples, max_length=1000, device="mps"
                        )
                elif name in ["PATE-CTGAN", "DP-CTGAN", "MWEM"]:
                    # DP models from snsynth use different sampling interface
                    actual_samples = (
                        min(gen_num_samples, 1000)
                        if self.small_training
                        else gen_num_samples
                    )
                    synthetic_data[name] = pd.DataFrame(
                        model.generate(actual_samples, self.train_data),
                        columns=self.train_data.columns,
                    )
                else:
                    actual_samples = (
                        min(gen_num_samples, 1000)
                        if self.small_training
                        else gen_num_samples
                    )
                    synthetic_data[name] = model.sample(num_rows=actual_samples)
            except TimeoutException:
                console.print(
                    f"[yellow]Data generation for {name} timed out after 1 hour."
                )
                synthetic_data[name] = pd.DataFrame(
                    columns=self.data.columns
                )  # Empty DataFrame with correct columns
            except Exception as e:
                console.print(f"[red]Error generating data for {name}: {str(e)}")
                synthetic_data[name] = pd.DataFrame(
                    columns=self.data.columns
                )  # Empty DataFrame with correct columns

            finally:
                signal.alarm(0)

            end_time = time.time()
            generation_time = end_time - start_time
            console.print(
                f"[bold green]Generation time for {name}: {generation_time:.2f} seconds"
            )
            if self.use_wandb:
                wandb.log({f"{name}_generation_time": generation_time})
        return synthetic_data

    def evaluate_models(
        self, synthetic_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for name, data in synthetic_data.items():
            console.print(f"Evaluating {name}...")
            start_time = time.time()
            try:
                if self.small_training and name == "GReaT":
                    results[name] = {
                        "quality_score": 1.0,
                        "column_shapes": 1.0,
                        "column_pair_trends": 1.0,
                        "ks_test": 1.0,
                        "correlation_similarity": 1.0,
                    }
                elif data.empty:
                    results[name] = {
                        "quality_score": np.nan,
                        "column_shapes": np.nan,
                        "column_pair_trends": np.nan,
                        "ks_test": np.nan,
                        "correlation_similarity": np.nan,
                    }
                else:
                    quality_report = QualityReport()
                    metadata_dict = self.metadata.to_dict()
                    quality_report.generate(
                        self.test_data, data, metadata_dict, verbose=self.verbose
                    )
                    results[name] = {
                        "quality_score": quality_report.get_score(),
                        "column_shapes": quality_report.get_details("Column Shapes"),
                        "column_pair_trends": quality_report.get_details(
                            "Column Pair Trends"
                        ),
                    }

                    results[name]["ks_test"] = self.kolmogorov_smirnov_test(
                        self.test_data, data
                    )
                    results[name]["correlation_similarity"] = (
                        self.correlation_similarity(self.test_data, data)
                    )
            except Exception as e:
                console.print(f"[red]Error evaluating {name}: {str(e)}")
                results[name] = {
                    "quality_score": np.nan,
                    "column_shapes": np.nan,
                    "column_pair_trends": np.nan,
                    "ks_test": np.nan,
                    "correlation_similarity": np.nan,
                }

            end_time = time.time()
            evaluation_time = end_time - start_time
            console.print(f"Evaluation time for {name}: {evaluation_time:.2f} seconds")
            if self.use_wandb:
                wandb.log({f"{name}_evaluation_time": evaluation_time})
                wandb.log({f"{name}_quality_score": results[name]["quality_score"]})
                wandb.log({f"{name}_ks_test": results[name]["ks_test"]})
                wandb.log(
                    {
                        f"{name}_correlation_similarity": results[name][
                            "correlation_similarity"
                        ]
                    }
                )
                wandb.log({f"{name}_column_shapes": results[name]["column_shapes"]})
                wandb.log(
                    {f"{name}_column_pair_trends": results[name]["column_pair_trends"]}
                )
        return results

    def display_results(self, results: Dict[str, Dict[str, float]], title: str):
        table = Table(title=title)
        table.add_column("Model", style="cyan")
        table.add_column("Quality Score", style="magenta")
        table.add_column("KS Test p-value", style="green")
        table.add_column("Correlation Similarity", style="yellow")
        # table.add_column("Column Shapes", style="blue")
        # table.add_column("Column Pair Trends", style="purple")

        for model, scores in results.items():
            table.add_row(
                model,
                f"{scores['quality_score']:.4f}"
                if not pd.isna(scores["quality_score"])
                else "-",
                f"{scores['ks_test']:.4f}" if not pd.isna(scores["ks_test"]) else "-",
                f"{scores['correlation_similarity']:.4f}"
                if not pd.isna(scores["correlation_similarity"])
                else "-",
                # f"{scores['column_shapes']:.4f}" if not pd.isna(scores['column_shapes']) else "-",
                # f"{scores['column_pair_trends']:.4f}" if not pd.isna(scores['column_pair_trends']) else "-"
            )

        console.print(table)

    def kolmogorov_smirnov_test(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> float:
        ks_scores = []
        for column in real_data.select_dtypes(include=[np.number]).columns:
            _, p_value = stats.ks_2samp(real_data[column], synthetic_data[column])
            ks_scores.append(p_value)
        return np.mean(ks_scores)

    def correlation_similarity(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> float:
        numeric_columns = real_data.select_dtypes(include=[np.number]).columns
        real_corr = real_data[numeric_columns].corr().values
        synthetic_corr = synthetic_data[numeric_columns].corr().values
        return np.corrcoef(real_corr.flatten(), synthetic_corr.flatten())[0, 1]

    def plot_distributions(self, synthetic_data: Dict[str, pd.DataFrame]):
        for feature in self.test_data.columns:
            plt.figure(figsize=(12, 6))
            plt.title(f"Distribution of {feature}", fontsize=16)

            if (
                self.test_data[feature].dtype == "object"
                or self.test_data[feature].dtype.name == "category"
            ):
                # Calculate ratios for categorical data
                original_ratios = self.test_data[feature].value_counts(normalize=True)
                synthetic_ratios = {
                    name: data[feature].value_counts(normalize=True)
                    for name, data in synthetic_data.items()
                }

                x = np.arange(len(original_ratios))
                width = 0.8 / (len(synthetic_data) + 1)

                plt.bar(
                    x,
                    original_ratios.values,
                    width,
                    label="Original",
                    color="black",
                    alpha=0.7,
                )
                for i, (name, ratios) in enumerate(synthetic_ratios.items(), start=1):
                    plt.bar(
                        x + i * width,
                        ratios.reindex(original_ratios.index).values,
                        width,
                        label=name,
                        alpha=0.5,
                    )

                plt.xticks(
                    x + width * (len(synthetic_data) / 2),
                    original_ratios.index,
                    rotation=45,
                    ha="right",
                )
            else:
                sns.kdeplot(
                    data=self.test_data,
                    x=feature,
                    label="Original",
                    color="black",
                    linewidth=2,
                )
                for name, data in synthetic_data.items():
                    sns.kdeplot(data=data, x=feature, label=name, fill=True, alpha=0.3)

            plt.xlabel(feature, fontsize=12)
            plt.ylabel(
                "Density"
                if self.test_data[feature].dtype != "object"
                else "Proportion",
                fontsize=12,
            )
            plt.legend(fontsize=10, loc="upper right")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    self.output_dir, "comparison", f"{feature}_distribution.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
            if self.use_wandb:
                wandb.log({f"{feature}_distribution": wandb.Image(plt)})
            plt.close()

    def plot_correlation_heatmaps(self, synthetic_data: Dict[str, pd.DataFrame]):
        numeric_columns = self.test_data.select_dtypes(include=[np.number]).columns

        # Calculate number of rows and columns needed for the subplot grid
        n_plots = len(synthetic_data) + 1  # +1 for original data
        n_cols = 3  # We'll keep 3 columns
        n_rows = (
            n_plots + n_cols - 1
        ) // n_cols  # Ceiling division to get number of rows

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
        axes = axes.flatten()

        # Plot original data correlation
        sns.heatmap(
            self.test_data[numeric_columns].corr(),
            ax=axes[0],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        axes[0].set_title("Original Data")

        # Plot synthetic data correlations
        for i, (name, data) in enumerate(synthetic_data.items(), start=1):
            sns.heatmap(
                data[numeric_columns].corr(),
                ax=axes[i],
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
            )
            axes[i].set_title(f"{name} Synthetic Data")

        # Hide any unused subplots
        for i in range(len(synthetic_data) + 1, len(axes)):
            axes[i].remove()

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "comparison", "correlation_heatmaps.png")
        )
        if self.use_wandb:
            wandb.log({"correlation_heatmaps": wandb.Image(plt)})
        plt.close()

    def plot_quality_scores(self, results: Dict[str, Dict[str, float]]):
        metrics = ["quality_score", "ks_test", "correlation_similarity"]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for i, metric in enumerate(metrics):
            scores = [results[model][metric] for model in results]
            axes[i].bar(results.keys(), scores)
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_ylabel("Score")
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "comparison", "metrics_comparison.png")
        )
        if self.use_wandb:
            wandb.log({"metrics_comparison": wandb.Image(plt)})
        plt.close()

    def plot_feature_comparison(self, synthetic_data: Dict[str, pd.DataFrame]):
        numeric_features = self.test_data.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            plt.figure(figsize=(10, 6))
            data = [self.test_data[feature]] + [
                synth_data[feature] for synth_data in synthetic_data.values()
            ]
            labels = ["Original"] + list(synthetic_data.keys())
            plt.boxplot(data, labels=labels)
            plt.title(f"Comparison of {feature} across models")
            plt.ylabel(feature)
            plt.savefig(
                os.path.join(self.output_dir, "comparison", f"{feature}_comparison.png")
            )
            if self.use_wandb:
                wandb.log({f"{feature}_comparison": wandb.Image(plt)})
            plt.close()

    def save_results(
        self,
        synthetic_data: Dict[str, pd.DataFrame],
        results: Dict[str, Dict[str, float]],
    ):
        for name, data in synthetic_data.items():
            data.to_csv(
                os.path.join(self.output_dir, name, f"{name}_synthetic_data.csv"),
                index=False,
            )

        json_safe_results = {}
        for model, model_results in results.items():
            json_safe_results[model] = {}
            for metric, value in model_results.items():
                if isinstance(value, pd.DataFrame):
                    json_safe_results[model][metric] = value.to_dict()
                elif isinstance(value, np.ndarray):
                    json_safe_results[model][metric] = value.tolist()
                else:
                    json_safe_results[model][metric] = value

        with open(
            os.path.join(self.output_dir, "comparison", "evaluation_results.json"), "w"
        ) as f:
            json.dump(json_safe_results, f, indent=4)

    def save_models(self):
        for name, model in self.models.items():
            try:
                if name == "GReaT":
                    model.save(os.path.join(self.output_dir, name, f"{name}_model.pkl"))
                elif name in ["PATE-CTGAN", "DP-CTGAN", "MWEM"]:
                    pass
                else:
                    model.save(os.path.join(self.output_dir, name, f"{name}_model.pkl"))
            except Exception as e:
                console.print(f"[red]Error saving {name} model: {str(e)}")

    def save_model(self, name: str, model, epoch: int):
        checkpoint_dir = os.path.join(self.output_dir, name, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pkl")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(model, f)

    def evaluate_privacy_metrics(
        self, synthetic_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate privacy metrics for DP methods."""
        privacy_results = {}

        for name, synth_data in synthetic_data.items():
            if name not in ["PATE-CTGAN", "DP-CTGAN"]:
                continue

            metrics = {
                "epsilon": 1.0,  # Default epsilon value used in training
                "delta": 1e-5,  # Default delta value used in training
                "privacy_risk_score": self.calculate_privacy_risk(
                    self.train_data, synth_data
                ),
                "attribute_disclosure_score": self.calculate_attribute_disclosure(
                    self.train_data, synth_data
                ),
                "wasserstein_distance": self.calculate_wasserstein_distances(
                    self.train_data, synth_data
                ),
            }

            privacy_results[name] = metrics

        return privacy_results

    def calculate_privacy_risk(
        self, real_data: pd.DataFrame, synth_data: pd.DataFrame
    ) -> float:
        """
        Calculate a simple privacy risk score based on nearest neighbor distance.
        Lower scores indicate better privacy.
        """
        risk_scores = []
        numeric_columns = real_data.select_dtypes(include=[np.number]).columns

        # Normalize the data
        real_normalized = (
            real_data[numeric_columns] - real_data[numeric_columns].mean()
        ) / real_data[numeric_columns].std()
        synth_normalized = (
            synth_data[numeric_columns] - synth_data[numeric_columns].mean()
        ) / synth_data[numeric_columns].std()

        # Sample for efficiency if datasets are large
        sample_size = min(1000, len(real_normalized))
        real_sample = real_normalized.sample(n=sample_size, random_state=42)
        synth_sample = synth_normalized.sample(n=sample_size, random_state=42)

        # Calculate minimum distances between real and synthetic samples
        for _, real_row in real_sample.iterrows():
            distances = np.linalg.norm(synth_sample - real_row, axis=1)
            risk_scores.append(np.min(distances))

        return float(np.mean(risk_scores))

    def calculate_attribute_disclosure(
        self, real_data: pd.DataFrame, synth_data: pd.DataFrame
    ) -> float:
        """
        Calculate attribute disclosure risk based on correlation preservation.
        Lower scores indicate better privacy.
        """
        numeric_columns = real_data.select_dtypes(include=[np.number]).columns

        real_corr = real_data[numeric_columns].corr()
        synth_corr = synth_data[numeric_columns].corr()

        # Calculate the difference in correlation matrices
        correlation_difference = np.abs(real_corr - synth_corr).mean().mean()

        return float(correlation_difference)

    def calculate_wasserstein_distances(
        self, real_data: pd.DataFrame, synth_data: pd.DataFrame
    ) -> float:
        """
        Calculate average Wasserstein distance across numerical features.
        Lower distances indicate more similarity (but potentially less privacy).
        """
        distances = []
        numeric_columns = real_data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            # Normalize the data
            real_col = (real_data[col] - real_data[col].mean()) / real_data[col].std()
            synth_col = (synth_data[col] - synth_data[col].mean()) / synth_data[
                col
            ].std()

            distance = wasserstein_distance(real_col, synth_col)
            distances.append(distance)

        return float(np.mean(distances))

    def display_privacy_results(self, privacy_results: Dict[str, Dict[str, float]]):
        """Display privacy evaluation results."""
        table = Table(title="Privacy Evaluation Results")
        table.add_column("Model", style="cyan")
        table.add_column("Epsilon", style="magenta")
        table.add_column("Delta", style="magenta")
        table.add_column("Privacy Risk Score", style="yellow")
        table.add_column("Attribute Disclosure", style="green")
        table.add_column("Wasserstein Distance", style="blue")

        for model, metrics in privacy_results.items():
            table.add_row(
                model,
                f"{metrics['epsilon']:.4f}",
                f"{metrics['delta']:.1e}",
                f"{metrics['privacy_risk_score']:.4f}",
                f"{metrics['attribute_disclosure_score']:.4f}",
                f"{metrics['wasserstein_distance']:.4f}",
            )

        console.print(table)

    def run_pipeline(self, gen_num_samples: int):
        with Progress(
            SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("1. [green]Running pipeline...", total=100)

            start_time = time.time()

            progress.update(task, advance=10, description="[green]Loading data...")
            self.load_data()

            # Remove or comment out the preprocess_data call
            # self.preprocess_data()

            progress.update(task, advance=10, description="[green]Preparing models...")
            self.prepare_models()

            progress.update(
                task,
                advance=10,
                description="[green]Score evaluation test vs validation data...",
            )
            self.dry_validation_run()

            progress.update(task, advance=30, description="[green]Training models...")
            self.train_models()
            progress.update(task, advance=30)  # Advance the progress bar after training

            progress.update(
                task, advance=20, description="[green]Generating synthetic data..."
            )
            synthetic_data = self.generate_synthetic_data(gen_num_samples)

            progress.update(task, advance=10, description="[green]Evaluating models...")
            results = self.evaluate_models(synthetic_data)
            self.display_results(results, "Final Evaluation Results")

            progress.update(
                task, advance=5, description="[green]Plotting distributions..."
            )
            self.plot_distributions(synthetic_data)

            progress.update(
                task, advance=5, description="[green]Plotting correlation heatmaps..."
            )
            self.plot_correlation_heatmaps(synthetic_data)

            progress.update(
                task, advance=5, description="[green]Plotting quality scores..."
            )
            self.plot_quality_scores(results)

            progress.update(task, advance=5, description="[green]Saving results...")
            self.save_results(synthetic_data, results)

            progress.update(task, advance=5, description="[green]Saving models...")
            self.save_models()

            progress.update(
                task, advance=5, description="[green]Evaluating privacy metrics..."
            )
            privacy_results = self.evaluate_privacy_metrics(synthetic_data)
            self.display_privacy_results(privacy_results)

            # Save privacy results along with other results
            if self.use_wandb:
                for model, metrics in privacy_results.items():
                    for metric_name, value in metrics.items():
                        wandb.log({f"{model}_{metric_name}": value})

            end_time = time.time()
            total_time = end_time - start_time
            console.print(
                f"[bold green]Total pipeline execution time: {total_time:.2f} seconds"
            )
            if self.use_wandb:
                wandb.log({"total_execution_time": total_time})


def main():
    parser = argparse.ArgumentParser(description="Run Synthetic Data Pipeline")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/sample_level_3_10k.csv",
        help="Path to the input data CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pipeline_output",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="default_experiment",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--gen_num_samples",
        type=int,
        default=10000,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--small_training",
        action="store_true",
        help="Use a small subset of data for testing",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Time limit for training in seconds",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "GaussianCopula",
            "CTGAN",
            "TVAE",
            "CopulaGAN",
            "GReaT",
            "PATE-CTGAN",  # Added
            "DP-CTGAN",  # Added
        ],
        help="Specify which models to train and evaluate",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    # epochs
    parser.add_argument(
        "--epochs",
        "-ep",
        type=int,
        default=300,
        help="Number of epochs to train the models",
    )
    parser.add_argument(
        "--epochs_great",
        "-ep_great",
        type=int,
        default=10,
        help="Number of epochs to train the GReaT model",
    )
    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(
            project="synthetic_data_pipeline",
            config=vars(args),
            name=args.experiment_name,
        )

    pipeline = SyntheticDataPipeline(
        args.data_path,
        args.output_dir,
        args.experiment_name,
        args.use_wandb,
        args.small_training,
        args.models,
        args.epochs,
        args.epochs_great,
        args.time_limit,
        args.verbose,
    )
    pipeline.run_pipeline(gen_num_samples=args.gen_num_samples)

    if args.use_wandb:
        # Create wandb Artifacts for plots
        for root, dirs, files in os.walk(
            os.path.join(args.output_dir, args.experiment_name, "comparison")
        ):
            for file in files:
                if file.endswith(".png"):
                    # Clean the filename to ensure it's a valid artifact name
                    clean_name = re.sub(r"[^a-zA-Z0-9\-_\.]", "_", file)
                    artifact = wandb.Artifact(name=f"plot_{clean_name}", type="plot")
                    artifact.add_file(os.path.join(root, file))
                    wandb.log_artifact(artifact)

        wandb.finish()


if __name__ == "__main__":
    main()
    # sample calls:
    # python -m notebooks.pipline.main --use_wandb --small_training --experiment_name small_test
    # python -m notebooks.pipline.main --models CTGAN GReaT --use_wandb --experiment_name ctgan_great_comparison
    # python -m notebooks.pipline.main  --experiment_name exp3 --use_wandb --gen_num_samples 10000
    # python -m notebooks.pipline.main  --experiment_name exp6_100k --use_wandb --gen_num_samples 10000 --data_path data/sample_level_3_100k.csv --models CTGAN CopulaGAN --epochs 150 --time_limit 7200 --verbose
    # python -m notebooks.pipline.main  --experiment_name exp5_100k_small --use_wandb --gen_num_samples 5000 --data_path data/sample_level_3_10k.csv --models CTGAN CopulaGAN --epochs 2 --time_limit 7200 --small_training
    # python -m notebooks.pipline.main  --experiment_name exp6_100k_great_lora --use_wandb --gen_num_samples 5000 --data_path data/sample_level_3_100k.csv --models GReaT -ep_great 100 --time_limit 30000

    # python -m notebooks.pipline.main --experiment_name exp_100k_all --use_wandb --gen_num_samples 10000 --data_path data/sample_level_3_100k.csv --models PATE-CTGAN DP-CTGAN --epochs 300 --epochs_great 10 --time_limit 3600 --verbose

    # python -m notebooks.pipline.main --experiment_name exp_100k_all2 --use_wandb --gen_num_samples 10000 --data_path data/sample_level_3_100k.csv --epochs 100 --epochs_great 10 --time_limit 3600 --verbose --output_dir pipeline_output/exp_100k_all2 --small_training
    # python -m notebooks.pipline.main --experiment_name exp_100k_all3 --use_wandb --gen_num_samples 10000 --data_path data/sample_level_3_100k.csv --epochs 100 --epochs_great 10 --time_limit 3600 --verbose --output_dir pipeline_output/exp_100k_all3
    # python -m notebooks.pipline.main --experiment_name exp_100k_all4 --use_wandb --gen_num_samples 10000 --data_path data/sample_level_3_100k.csv --epochs 100 --epochs_great 10 --time_limit 3600 --verbose --output_dir pipeline_output/exp_100k_all4
    #
