"""
AML 360º - Graph Analyzer
Advanced graph analytics for transaction network analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any
import networkx as nx
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

# Graph analytics libraries
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False

try:
    from community import community_louvain
    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False

class GraphAnalyzer:
    """
    Advanced graph analytics for AML transaction networks
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for transaction flows
        self.undirected_graph = nx.Graph()  # Undirected for community detection
        self.entity_metadata = {}
        self.communities = {}
        self.centrality_cache = {}
        
    def build_graph_from_transactions(self, transactions_df: pd.DataFrame) -> None:
        """
        Build transaction graph from DataFrame
        
        Expected columns: payer_id, payee_id, amount, timestamp, transaction_id
        """
        
        # Clear existing graph
        self.graph.clear()
        self.undirected_graph.clear()
        
        print(f"Building graph from {len(transactions_df):,} transactions...")
        
        # Group transactions by payer-payee pairs
        edge_data = transactions_df.groupby(['payer_id', 'payee_id']).agg({
            'amount': ['sum', 'mean', 'count', 'std'],
            'timestamp': ['min', 'max'],
            'transaction_id': 'count'
        }).round(2)
        
        # Flatten column names
        edge_data.columns = [
            'total_amount', 'avg_amount', 'transaction_count', 'amount_std',
            'first_transaction', 'last_transaction', 'tx_count_check'
        ]
        
        edge_data = edge_data.reset_index()
        edge_data['amount_std'] = edge_data['amount_std'].fillna(0)
        
        # Add edges to directed graph
        for _, row in edge_data.iterrows():
            payer = row['payer_id']
            payee = row['payee_id']
            
            # Skip self-loops
            if payer == payee:
                continue
            
            # Calculate edge weight (log of total amount for better scaling)
            weight = np.log1p(row['total_amount'])
            
            # Add edge with comprehensive attributes
            self.graph.add_edge(
                payer, payee,
                weight=weight,
                total_amount=row['total_amount'],
                avg_amount=row['avg_amount'],
                transaction_count=row['transaction_count'],
                amount_std=row['amount_std'],
                first_transaction=row['first_transaction'],
                last_transaction=row['last_transaction'],
                # Risk indicators
                velocity_score=row['transaction_count'] / 30,  # Transactions per day assumption
                amount_concentration=row['amount_std'] / (row['avg_amount'] + 1e-6),
                suspicious_timing=self._calculate_timing_score(
                    row['first_transaction'], row['last_transaction'], row['transaction_count']
                )
            )
            
            # Add to undirected graph for community detection (combine weights)
            if self.undirected_graph.has_edge(payer, payee):
                self.undirected_graph[payer][payee]['weight'] += weight
                self.undirected_graph[payer][payee]['total_amount'] += row['total_amount']
            else:
                self.undirected_graph.add_edge(
                    payer, payee,
                    weight=weight,
                    total_amount=row['total_amount']
                )
        
        print(f"Graph built: {self.graph.number_of_nodes():,} nodes, {self.graph.number_of_edges():,} edges")
        
        # Calculate entity metadata
        self._calculate_entity_metadata()
        
        # Detect communities
        self._detect_communities()
    
    def _calculate_timing_score(self, first_ts, last_ts, tx_count):
        """Calculate suspicious timing score"""
        if pd.isna(first_ts) or pd.isna(last_ts) or tx_count <= 1:
            return 0
        
        # Time span in days
        time_span = (last_ts - first_ts).total_seconds() / (24 * 3600)
        
        if time_span <= 0:
            return 1  # All transactions in same day
        
        # Transactions per day
        tx_per_day = tx_count / time_span
        
        # Higher score for more concentrated timing
        return min(tx_per_day / 10, 1)  # Normalize to 0-1
    
    def _calculate_entity_metadata(self):
        """Calculate metadata for each entity"""
        print("Calculating entity metadata...")
        
        self.entity_metadata = {}
        
        for node in self.graph.nodes():
            # Basic connectivity
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            total_degree = in_degree + out_degree
            
            # Financial flows
            in_amount = sum([
                self.graph[pred][node].get('total_amount', 0) 
                for pred in self.graph.predecessors(node)
            ])
            out_amount = sum([
                self.graph[node][succ].get('total_amount', 0) 
                for succ in self.graph.successors(node)
            ])
            
            # Transaction counts
            in_transactions = sum([
                self.graph[pred][node].get('transaction_count', 0) 
                for pred in self.graph.predecessors(node)
            ])
            out_transactions = sum([
                self.graph[node][succ].get('transaction_count', 0) 
                for succ in self.graph.successors(node)
            ])
            
            # Risk indicators
            flow_imbalance = abs(in_amount - out_amount) / (in_amount + out_amount + 1e-6)
            degree_centrality = total_degree / (self.graph.number_of_nodes() - 1)
            
            self.entity_metadata[node] = {
                'in_degree': in_degree,
                'out_degree': out_degree,
                'total_degree': total_degree,
                'in_amount': in_amount,
                'out_amount': out_amount,
                'net_flow': in_amount - out_amount,
                'total_flow': in_amount + out_amount,
                'in_transactions': in_transactions,
                'out_transactions': out_transactions,
                'total_transactions': in_transactions + out_transactions,
                'flow_imbalance': flow_imbalance,
                'degree_centrality': degree_centrality,
                'avg_in_amount': in_amount / (in_transactions + 1e-6),
                'avg_out_amount': out_amount / (out_transactions + 1e-6)
            }
    
    def _detect_communities(self):
        """Detect communities using multiple algorithms"""
        print("Detecting communities...")
        
        if self.undirected_graph.number_of_nodes() < 3:
            print("Not enough nodes for community detection")
            return
        
        self.communities = {}
        
        # Louvain community detection (if available)
        if COMMUNITY_AVAILABLE and self.undirected_graph.number_of_edges() > 0:
            try:
                partition = community_louvain.best_partition(self.undirected_graph, weight='weight')
                self.communities['louvain'] = partition
                print(f"Louvain: {len(set(partition.values()))} communities detected")
            except Exception as e:
                print(f"Louvain detection failed: {e}")
        
        # NetworkX built-in community detection
        try:
            # Greedy modularity communities
            communities_gen = nx.community.greedy_modularity_communities(
                self.undirected_graph, weight='weight'
            )
            communities_list = list(communities_gen)
            
            # Convert to node->community mapping
            greedy_partition = {}
            for i, community in enumerate(communities_list):
                for node in community:
                    greedy_partition[node] = i
            
            self.communities['greedy_modularity'] = greedy_partition
            print(f"Greedy modularity: {len(communities_list)} communities detected")
        except Exception as e:
            print(f"Greedy modularity detection failed: {e}")
        
        # Label propagation
        try:
            communities_gen = nx.community.label_propagation_communities(self.undirected_graph)
            communities_list = list(communities_gen)
            
            label_partition = {}
            for i, community in enumerate(communities_list):
                for node in community:
                    label_partition[node] = i
            
            self.communities['label_propagation'] = label_partition
            print(f"Label propagation: {len(communities_list)} communities detected")
        except Exception as e:
            print(f"Label propagation detection failed: {e}")
    
    def calculate_centrality_measures(self, force_recalculate: bool = False) -> Dict[str, Dict]:
        """Calculate various centrality measures"""
        
        if not force_recalculate and self.centrality_cache:
            return self.centrality_cache
        
        print("Calculating centrality measures...")
        
        centralities = {}
        
        try:
            # Degree centrality (fast)
            centralities['degree'] = nx.degree_centrality(self.undirected_graph)
            centralities['in_degree'] = nx.in_degree_centrality(self.graph)
            centralities['out_degree'] = nx.out_degree_centrality(self.graph)
            
            print("✅ Degree centralities calculated")
        except Exception as e:
            print(f"❌ Degree centrality failed: {e}")
        
        try:
            # Betweenness centrality (expensive for large graphs)
            if self.graph.number_of_nodes() < 5000:  # Only for smaller graphs
                centralities['betweenness'] = nx.betweenness_centrality(
                    self.undirected_graph, weight='weight', k=min(1000, self.graph.number_of_nodes())
                )
                print("✅ Betweenness centrality calculated")
            else:
                print("⏭️ Skipping betweenness centrality (graph too large)")
        except Exception as e:
            print(f"❌ Betweenness centrality failed: {e}")
        
        try:
            # Eigenvector centrality
            centralities['eigenvector'] = nx.eigenvector_centrality(
                self.undirected_graph, weight='weight', max_iter=1000, tol=1e-6
            )
            print("✅ Eigenvector centrality calculated")
        except Exception as e:
            print(f"❌ Eigenvector centrality failed: {e}")
            # Fallback to degree centrality
            centralities['eigenvector'] = centralities.get('degree', {})
        
        try:
            # PageRank
            centralities['pagerank'] = nx.pagerank(
                self.graph, weight='weight', alpha=0.85, max_iter=1000, tol=1e-6
            )
            print("✅ PageRank calculated")
        except Exception as e:
            print(f"❌ PageRank failed: {e}")
        
        try:
            # Closeness centrality (expensive)
            if self.graph.number_of_nodes() < 2000:
                centralities['closeness'] = nx.closeness_centrality(
                    self.undirected_graph, distance='weight'
                )
                print("✅ Closeness centrality calculated")
            else:
                print("⏭️ Skipping closeness centrality (graph too large)")
        except Exception as e:
            print(f"❌ Closeness centrality failed: {e}")
        
        # Cache results
        self.centrality_cache = centralities
        print(f"✅ Calculated {len(centralities)} centrality measures")
        
        return centralities
    
    def analyze_entity(self, entity_id: str) -> Dict[str, Any]:
        """Comprehensive analysis of a specific entity"""
        
        if entity_id not in self.graph:
            return {
                'error': f'Entity {entity_id} not found in graph',
                'exists': False
            }
        
        # Basic metadata
        metadata = self.entity_metadata.get(entity_id, {})
        
        # Centrality scores
        centralities = self.calculate_centrality_measures()
        entity_centralities = {}
        
        for measure, values in centralities.items():
            entity_centralities[measure] = values.get(entity_id, 0)
        
        # Community memberships
        entity_communities = {}
        for method, partition in self.communities.items():
            entity_communities[method] = partition.get(entity_id, -1)
        
        # Network neighborhood
        neighbors = {
            'predecessors': list(self.graph.predecessors(entity_id)),
            'successors': list(self.graph.successors(entity_id)),
            'all_neighbors': list(nx.all_neighbors(self.undirected_graph, entity_id))
        }
        
        # Risk assessment
        risk_score = self._calculate_entity_risk_score(entity_id, metadata, entity_centralities)
        
        # Suspicious patterns
        patterns = self._detect_suspicious_patterns(entity_id)
        
        return {
            'entity_id': entity_id,
            'exists': True,
            'metadata': metadata,
            'centralities': entity_centralities,
            'communities': entity_communities,
            'neighbors': neighbors,
            'risk_score': risk_score,
            'suspicious_patterns': patterns,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _calculate_entity_risk_score(self, entity_id: str, metadata: Dict, centralities: Dict) -> Dict[str, float]:
        """Calculate comprehensive risk score for entity"""
        
        risk_components = {}
        
        # Connectivity risk (high centrality = higher risk)
        risk_components['centrality_risk'] = min(
            centralities.get('degree', 0) * 2 +
            centralities.get('betweenness', 0) * 3 +
            centralities.get('pagerank', 0) * 100,
            1.0
        )
        
        # Volume risk (very high or very low volumes)
        total_flow = metadata.get('total_flow', 0)
        if total_flow > 0:
            # Log-based normalization
            volume_score = min(np.log10(total_flow) / 6, 1.0)  # Normalize to 0-1
            risk_components['volume_risk'] = volume_score
        else:
            risk_components['volume_risk'] = 0
        
        # Flow imbalance risk
        risk_components['imbalance_risk'] = metadata.get('flow_imbalance', 0)
        
        # Velocity risk (many transactions in short time)
        total_tx = metadata.get('total_transactions', 0)
        if total_tx > 0:
            # Assume transactions over 30 days, calculate daily rate
            velocity_score = min(total_tx / 30 / 50, 1.0)  # Normalize: 50 tx/day = max risk
            risk_components['velocity_risk'] = velocity_score
        else:
            risk_components['velocity_risk'] = 0
        
        # Structural risk (unusual network position)
        degree_ratio = metadata.get('out_degree', 0) / max(metadata.get('in_degree', 1), 1)
        if degree_ratio > 10 or degree_ratio < 0.1:  # Very unbalanced
            risk_components['structural_risk'] = 0.8
        else:
            risk_components['structural_risk'] = 0.1
        
        # Composite risk score (weighted average)
        weights = {
            'centrality_risk': 0.25,
            'volume_risk': 0.25,
            'imbalance_risk': 0.20,
            'velocity_risk': 0.20,
            'structural_risk': 0.10
        }
        
        composite_risk = sum(
            risk_components[component] * weights[component]
            for component in weights
            if component in risk_components
        )
        
        risk_components['composite_risk'] = min(composite_risk, 1.0)
        
        return risk_components
    
    def _detect_suspicious_patterns(self, entity_id: str) -> List[Dict[str, Any]]:
        """Detect suspicious patterns for an entity"""
        
        patterns = []
        
        # Pattern 1: Hub (many connections)
        degree = self.undirected_graph.degree(entity_id)
        if degree > 100:  # Threshold for hub detection
            patterns.append({
                'pattern': 'hub',
                'description': f'Entity has {degree} connections (potential hub)',
                'severity': 'medium' if degree < 500 else 'high',
                'score': min(degree / 1000, 1.0)
            })
        
        # Pattern 2: Bridge (high betweenness centrality)
        centralities = self.calculate_centrality_measures()
        betweenness = centralities.get('betweenness', {}).get(entity_id, 0)
        
        if betweenness > 0.01:  # High betweenness threshold
            patterns.append({
                'pattern': 'bridge',
                'description': f'Entity acts as bridge in network (betweenness: {betweenness:.4f})',
                'severity': 'high',
                'score': betweenness
            })
        
        # Pattern 3: Rapid fire (many transactions in short time)
        metadata = self.entity_metadata.get(entity_id, {})
        total_tx = metadata.get('total_transactions', 0)
        
        if total_tx > 1000:  # High velocity threshold
            patterns.append({
                'pattern': 'high_velocity',
                'description': f'High transaction velocity: {total_tx} transactions',
                'severity': 'medium',
                'score': min(total_tx / 5000, 1.0)
            })
        
        # Pattern 4: Flow anomaly (large imbalance)
        flow_imbalance = metadata.get('flow_imbalance', 0)
        if flow_imbalance > 0.8:
            patterns.append({
                'pattern': 'flow_imbalance',
                'description': f'Unusual flow pattern (imbalance: {flow_imbalance:.2f})',
                'severity': 'medium',
                'score': flow_imbalance
            })
        
        # Pattern 5: Isolated community (small community membership)
        for method, partition in self.communities.items():
            community_id = partition.get(entity_id, -1)
            if community_id != -1:
                # Count community size
                community_size = sum(1 for v in partition.values() if v == community_id)
                if community_size < 5:  # Very small community
                    patterns.append({
                        'pattern': 'isolated_community',
                        'description': f'Member of small isolated community (size: {community_size})',
                        'severity': 'low',
                        'score': 1.0 / community_size
                    })
                    break  # Only report once
        
        return patterns
    
    def get_community_info(self, entity_id: str) -> Dict[str, Any]:
        """Get detailed community information for an entity"""
        
        if entity_id not in self.graph:
            return {'error': f'Entity {entity_id} not found'}
        
        community_info = {}
        
        for method, partition in self.communities.items():
            community_id = partition.get(entity_id, -1)
            
            if community_id == -1:
                continue
            
            # Find all members of this community
            community_members = [
                node for node, comm in partition.items() 
                if comm == community_id
            ]
            
            # Calculate community statistics
            community_stats = self._calculate_community_stats(community_members)
            
            community_info[method] = {
                'community_id': community_id,
                'size': len(community_members),
                'members': community_members[:10],  # Limit to first 10
                'total_members': len(community_members),
                'statistics': community_stats
            }
        
        return {
            'entity_id': entity_id,
            'communities': community_info
        }
    
    def _calculate_community_stats(self, members: List[str]) -> Dict[str, float]:
        """Calculate statistics for a community"""
        
        if not members:
            return {}
        
        # Aggregate metadata for community members
        total_flow = 0
        total_transactions = 0
        avg_centralities = defaultdict(float)
        
        centralities = self.calculate_centrality_measures()
        
        for member in members:
            metadata = self.entity_metadata.get(member, {})
            total_flow += metadata.get('total_flow', 0)
            total_transactions += metadata.get('total_transactions', 0)
            
            # Average centralities
            for measure, values in centralities.items():
                avg_centralities[measure] += values.get(member, 0)
        
        # Calculate averages
        n_members = len(members)
        for measure in avg_centralities:
            avg_centralities[measure] /= n_members
        
        return {
            'total_flow': total_flow,
            'avg_flow_per_member': total_flow / n_members,
            'total_transactions': total_transactions,
            'avg_transactions_per_member': total_transactions / n_members,
            'avg_centralities': dict(avg_centralities)
        }
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Get overall network summary statistics"""
        
        # Basic network stats
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        density = nx.density(self.graph)
        
        # Connected components
        n_components = nx.number_weakly_connected_components(self.graph)
        largest_component_size = len(max(
            nx.weakly_connected_components(self.graph), key=len
        )) if n_components > 0 else 0
        
        # Community stats
        community_stats = {}
        for method, partition in self.communities.items():
            n_communities = len(set(partition.values()))
            avg_community_size = len(partition) / n_communities if n_communities > 0 else 0
            
            community_stats[method] = {
                'n_communities': n_communities,
                'avg_community_size': avg_community_size
            }
        
        # Risk distribution
        if self.entity_metadata:
            risk_scores = []
            for entity_id in self.entity_metadata:
                entity_analysis = self.analyze_entity(entity_id)
                composite_risk = entity_analysis.get('risk_score', {}).get('composite_risk', 0)
                risk_scores.append(composite_risk)
            
            risk_distribution = {
                'mean_risk': np.mean(risk_scores),
                'std_risk': np.std(risk_scores),
                'high_risk_entities': sum(1 for r in risk_scores if r > 0.7),
                'low_risk_entities': sum(1 for r in risk_scores if r < 0.3)
            }
        else:
            risk_distribution = {}
        
        return {
            'network_size': {
                'nodes': n_nodes,
                'edges': n_edges,
                'density': density
            },
            'connectivity': {
                'components': n_components,
                'largest_component_size': largest_component_size,
                'connectivity_ratio': largest_component_size / n_nodes if n_nodes > 0 else 0
            },
            'communities': community_stats,
            'risk_distribution': risk_distribution,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    
    print("🕸️ AML Graph Analyzer - Testing")
    
    # Create sample transaction data
    np.random.seed(42)
    n_transactions = 10000
    n_entities = 1000
    
    entities = [f'entity_{i:04d}' for i in range(n_entities)]
    
    # Generate synthetic transactions with some clustering
    transactions = []
    for i in range(n_transactions):
        # 80% random, 20% clustered around hubs
        if np.random.random() < 0.8:
            payer = np.random.choice(entities)
            payee = np.random.choice(entities)
        else:
            # Create hub-like patterns
            hub = np.random.choice(entities[:50])  # Top 50 entities as hubs
            if np.random.random() < 0.5:
                payer = hub
                payee = np.random.choice(entities[50:])
            else:
                payer = np.random.choice(entities[50:])
                payee = hub
        
        transactions.append({
            'transaction_id': f'tx_{i:08d}',
            'payer_id': payer,
            'payee_id': payee,
            'amount': np.random.lognormal(8, 1.5),
            'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 30))
        })
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Initialize and test analyzer
    analyzer = GraphAnalyzer()
    analyzer.build_graph_from_transactions(df)
    
    # Test analysis
    print("\n🔍 Testing entity analysis...")
    test_entity = entities[0]
    analysis = analyzer.analyze_entity(test_entity)
    
    print(f"Analysis for {test_entity}:")
    print(f"  Risk score: {analysis['risk_score']['composite_risk']:.3f}")
    print(f"  Degree centrality: {analysis['centralities'].get('degree', 0):.3f}")
    print(f"  PageRank: {analysis['centralities'].get('pagerank', 0):.6f}")
    print(f"  Suspicious patterns: {len(analysis['suspicious_patterns'])}")
    
    # Test network summary
    print("\n📊 Network summary:")
    summary = analyzer.get_network_summary()
    print(f"  Nodes: {summary['network_size']['nodes']:,}")
    print(f"  Edges: {summary['network_size']['edges']:,}")
    print(f"  Components: {summary['connectivity']['components']}")
    print(f"  Mean risk: {summary['risk_distribution'].get('mean_risk', 0):.3f}")
    
    print("\n✅ Graph analyzer testing complete!")
