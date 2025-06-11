#!/usr/bin/env python3

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Set Java environment - COPY FROM YOUR JUPYTER NOTEBOOK
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-8.0.452.9-hotspot"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 8g pyspark-shell"

import findspark
findspark.init()  # Initialize findspark before importing pyspark
findspark.init()  # Initialize findspark before importing pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

# Initialize Spark (use simpler approach like your Jupyter notebook)
def create_fresh_spark_context(app_name):
    """Create a fresh Spark context using same approach as Jupyter"""
    # Stop any existing context
    try:
        spark = SparkSession.getActiveSession()
        if spark:
            spark.stop()
    except:
        pass
    
    # Wait for cleanup
    time.sleep(3)
    
    # Use same config as your working Jupyter notebook
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
    
    return spark

# Data sampling function
def create_independent_sample(df, fraction, seed=None):
    """Create completely independent random sample"""
    if seed is None:
        seed = int(time.time() * 1000) % 2**31  # Random seed each time
    
    sample_df = df.sample(fraction=fraction, seed=seed)
    return sample_df

# Graph building function (simplified for testing)
def build_test_graph(df, threshold=2):
    """Build graph for RDD testing - optimized for memory"""
    # Build user-book relationships
    user_books = df.select("user_id_int", "book_id")
    
    # Create book pairs through shared users
    user_aggs = user_books.groupBy("user_id_int").agg(F.collect_set("book_id").alias("books"))
    
    # Generate all book pairs for each user
    book_pairs = user_aggs.select(
        F.explode("books").alias("src"),
        "books"
    ).withColumn("dst", F.explode("books")) \
     .filter(F.col("src") < F.col("dst"))  # Avoid duplicates and self-loops
    
    # Count co-occurrences and filter by threshold
    edges_df = book_pairs.groupBy("src", "dst").count() \
        .filter(F.col("count") >= threshold) \
        .select("src", "dst")
    
    # Make bidirectional (undirected graph)
    edges_bidirectional = edges_df.union(
        edges_df.select(F.col("dst").alias("src"), F.col("src").alias("dst"))
    )
    
    # Get graph statistics
    num_edges = edges_bidirectional.count()
    all_nodes_df = edges_bidirectional.select("src").union(edges_bidirectional.select("dst")).distinct()
    num_nodes = all_nodes_df.count()
    
    # Convert to RDD format for PageRank
    edges_rdd = edges_bidirectional.rdd.map(lambda row: (row["src"], row["dst"]))
    
    return {
        "edges_rdd": edges_rdd,
        "nodes_count": num_nodes,
        "edges_count": num_edges
    }

# Your exact RDD PageRank function with silent mode and partitioning
def pagerank_rdd_clean(edges_rdd, nodes_count, damping_factor=0.85, max_iter=100, tolerance=1e-6, silent=True):
    """
    Clean PageRank RDD implementation with silent mode
    """
    # CRITICAL: Repartition for balanced workload
    edges_rdd_partitioned = edges_rdd.repartition(4).cache()
    edges_rdd_partitioned.count()  # Force materialization
    
    # Build adjacency list
    adjacency_rdd = edges_rdd_partitioned.groupByKey().mapValues(list).cache()
    all_nodes = edges_rdd_partitioned.flatMap(lambda x: [x[0], x[1]]).distinct().collect()
    
    # Initialize ranks uniformly
    ranks = {node: 1.0 / len(all_nodes) for node in all_nodes}
    
    if not silent:
        print(f"    Starting PageRank: {nodes_count} nodes, {len(all_nodes)} in RDD")
    
    for iteration in range(max_iter):
        old_ranks = ranks.copy()
        
        # Broadcast current ranks
        ranks_bc = edges_rdd_partitioned.context.broadcast(ranks)
        
        # Calculate rank contributions using RDD operations
        contributions = adjacency_rdd.flatMap(
            lambda node_neighbors: [
                (neighbor, damping_factor * ranks_bc.value[node_neighbors[0]] / len(node_neighbors[1]))
                for neighbor in node_neighbors[1]
            ]
        ).reduceByKey(lambda a, b: a + b).collectAsMap()
        
        # Update ranks with random jump probability
        random_jump = (1 - damping_factor) / len(all_nodes)
        ranks = {node: random_jump + contributions.get(node, 0) for node in all_nodes}
        
        # Check convergence
        diff = sum((ranks[node] - old_ranks[node])**2 for node in all_nodes)**0.5
        
        # Print progress (only if not silent and at intervals)
        if not silent and (iteration % 10 == 0 or diff < tolerance):
            total_sum = sum(ranks.values())
            print(f"    Iteration {iteration + 1}, L2 Distance: {diff:.6f}, Sum: {total_sum:.6f}")
        
        # Clean up broadcast variable
        ranks_bc.unpersist()
        
        # Check convergence
        if diff < tolerance:
            if not silent:
                print(f"    Converged after {iteration + 1} iterations")
            break
    
    # Clean up cached RDDs
    adjacency_rdd.unpersist()
    edges_rdd_partitioned.unpersist()
    
    # Return sorted results and iteration count
    return sorted(ranks.items(), key=lambda x: -x[1]), (iteration + 1)

# Scaling test for RDD PageRank
def run_pagerank_scaling_test(edges_rdd, nodes_count):
    """
    Test RDD PageRank function for scaling performance
    """
    start_time = time.time()
    
    # Run PageRank (silent mode)
    results, iterations = pagerank_rdd_clean(
        edges_rdd, 
        nodes_count,
        damping_factor=0.85,
        max_iter=100,
        tolerance=1e-6,
        silent=True
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        "iterations": iterations,
        "total_time": total_time,
        "time_per_iteration": total_time / iterations if iterations > 0 else 0,
        "converged": iterations < 100,
        "final_rank_sum": sum(rank for _, rank in results) if results else 0
    }

# Main scaling test function
def run_scaling_test(data_path, output_path="rdd_pagerank_scaling_results.csv"):
    """
    Run complete scaling test across different data fractions
    """
    # Test fractions from small to full dataset
    fractions = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    
    results = []
    
    print("RDD PAGERANK SCALING TEST")
    print("Testing classic RDD PageRank with proper partitioning")
    print("Each test uses a fresh Spark context to avoid memory accumulation")
    
    for i, fraction in enumerate(fractions):
        print(f"\n[{i+1}/{len(fractions)}] Testing {fraction:.1%} of data...")
        
        # Create fresh Spark context for each test
        spark = create_fresh_spark_context(f"RDD_PageRank_Scaling_{fraction:.1%}")
        
        try:
            # Load data
            print(f"  Loading data from {data_path}...")
            if data_path.endswith('.pkl'):
                pandas_df = pd.read_pickle(data_path)
                df = spark.createDataFrame(pandas_df)
            else:
                df = spark.read.csv(data_path, header=True, inferSchema=True)
            
            # Create sample
            sample_df = create_independent_sample(df, fraction)
            sample_count = sample_df.count()
            print(f"  Sample size: {sample_count:,} reviews")
            
            # Build graph
            print(f"  Building graph...")
            graph_data = build_test_graph(sample_df, threshold=2)
            print(f"  Graph: {graph_data['nodes_count']:,} nodes, {graph_data['edges_count']:,} edges")
            
            # Check if graph is reasonable size
            if graph_data['edges_count'] == 0:
                print(f"  No edges found - sample too small or threshold too high")
                results.append({
                    'fraction': fraction,
                    'sample_reviews': sample_count,
                    'nodes': 0,
                    'edges': 0,
                    'pagerank_iterations': 0,
                    'pagerank_total_time': 0,
                    'pagerank_time_per_iteration': 0,
                    'converged': False,
                    'error': 'No edges in graph'
                })
                continue
            
            # Run PageRank (this is what we're timing)
            print(f"  Running RDD PageRank...")
            start_pr = time.time()
            pr_results = run_pagerank_scaling_test(
                graph_data['edges_rdd'], 
                graph_data['nodes_count']
            )
            end_pr = time.time()
            
            # Store results
            result = {
                'fraction': fraction,
                'sample_reviews': sample_count,
                'nodes': graph_data['nodes_count'],
                'edges': graph_data['edges_count'],
                'pagerank_iterations': pr_results['iterations'],
                'pagerank_total_time': pr_results['total_time'],
                'pagerank_time_per_iteration': pr_results['time_per_iteration'],
                'converged': pr_results['converged'],
                'rank_sum': pr_results['final_rank_sum']
            }
            
            results.append(result)
            
            # Print success
            print(f"  SUCCESS: {pr_results['iterations']} iterations, {pr_results['total_time']:.2f}s total, {pr_results['time_per_iteration']:.4f}s/iter")
            
            # Early termination if getting too slow
            if pr_results['total_time'] > 300:  # 5 minutes
                print(f"  PageRank taking too long ({pr_results['total_time']:.1f}s), consider stopping here")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            # Record error but continue
            results.append({
                'fraction': fraction,
                'sample_reviews': 0,
                'nodes': 0,
                'edges': 0,
                'pagerank_iterations': 0,
                'pagerank_total_time': 0,
                'pagerank_time_per_iteration': 0,
                'converged': False,
                'error': str(e)
            })
        
        finally:
            # Always clean up Spark context
            try:
                spark.stop()
            except:
                pass
            time.sleep(3)  # Give Spark time to fully shut down
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return results_df

# Plotting function
def plot_scaling_results(results_df, save_plots=True):
    """
    Create scaling analysis plot
    """
    # Filter out error cases
    clean_results = results_df[
        (results_df['nodes'] > 0) & 
        (results_df['pagerank_total_time'] > 0)
    ].copy()
    
    if len(clean_results) == 0:
        print("No valid results to plot!")
        return
    
    print(f"Plotting {len(clean_results)} successful test points")
    
    # Only plot time per iteration vs edges
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Time per iteration vs Edges
    ax.plot(clean_results['edges'], clean_results['pagerank_time_per_iteration'], 'o-', color='blue', linewidth=3, markersize=10)
    ax.set_xlabel('Number of Edges', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time per Iteration (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('RDD PageRank Scaling: Time per Iteration vs Edges', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add scaling trend line for reference
    z = np.polyfit(clean_results['edges'], clean_results['pagerank_time_per_iteration'], 1)
    p = np.poly1d(z)
    ax.plot(clean_results['edges'], p(clean_results['edges']), "--", color='red', alpha=0.7, linewidth=2, 
           label=f'Linear trend: {z[0]:.2e}x + {z[1]:.2f}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_plots:
        plot_filename = 'rdd_pagerank_scaling_analysis.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_filename}")
    
    plt.show()
    
    # Print summary statistics
    print("\nRDD PAGERANK SCALING ANALYSIS SUMMARY")
    print(f"Successful tests: {len(clean_results)}")
    print(f"Graph size range: {clean_results['edges'].min():,} - {clean_results['edges'].max():,} edges")
    print(f"Node range: {clean_results['nodes'].min():,} - {clean_results['nodes'].max():,} nodes")
    print(f"Time per iteration: {clean_results['pagerank_time_per_iteration'].min():.4f} - {clean_results['pagerank_time_per_iteration'].max():.4f} seconds")
    print(f"Total runtime range: {clean_results['pagerank_total_time'].min():.2f} - {clean_results['pagerank_total_time'].max():.2f} seconds")
    print(f"Iterations range: {clean_results['pagerank_iterations'].min()} - {clean_results['pagerank_iterations'].max()}")
    print(f"Average time per iteration: {clean_results['pagerank_time_per_iteration'].mean():.4f} seconds")
    print(f"Convergence rate: {clean_results['converged'].mean()*100:.1f}%")
    
    # Calculate scaling factors
    if len(clean_results) > 1:
        edge_ratio = clean_results['edges'].max() / clean_results['edges'].min()
        time_ratio = clean_results['pagerank_time_per_iteration'].max() / clean_results['pagerank_time_per_iteration'].min()
        print(f"\nScaling analysis:")
        print(f"Edge count increased by: {edge_ratio:.1f}x")
        print(f"Time per iteration increased by: {time_ratio:.1f}x")
        print(f"Scaling efficiency: {edge_ratio/time_ratio:.2f} (higher = better)")

# Usage and main execution
if __name__ == "__main__":
    # Set your data path here
    DATA_PATH = "C:\\Users\\diana\\Downloads\\pandas_data\\final_data.csv"  # Change this to your actual path

    # Verify file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please update DATA_PATH to point to your data file")
        exit(1)
    
    print("Starting RDD PageRank Scaling Test...")
    print(f"Data source: {DATA_PATH}")
    print("This will test multiple data fractions with fresh Spark contexts")
    print("Each test is completely independent to avoid memory accumulation")
    print("Running in silent mode for clean output")
    
    # Run the scaling test
    results = run_scaling_test(DATA_PATH)
    
    # Show summary
    successful_tests = len(results[results['nodes'] > 0])
    total_tests = len(results)
    print(f"\nCompleted: {successful_tests}/{total_tests} tests successful")
    
    if successful_tests > 0:
        # Plot results
        plot_scaling_results(results)
        
        # Display results table
        print("\nDetailed Results:")
        display_cols = ['fraction', 'nodes', 'edges', 'pagerank_iterations', 
                       'pagerank_time_per_iteration', 'pagerank_total_time', 'converged']
        print(results[display_cols].round(4))
    else:
        print("No successful tests - check data path and sample sizes")
        print("\nFailed results:")
        print(results[['fraction', 'error']] if 'error' in results.columns else results)