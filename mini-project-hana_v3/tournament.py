"""
Tournament and evaluation script for Hanamikoji agents.
Use this to test your agents against others and gather statistics.
"""

from medium_hana import Game, RandomAgent
from medium_hana.example_agents import GreedyAgent, BalancedAgent, CardCountingAgent
from typing import List, Type, Dict
import time


class AgentEvaluator:
    """Evaluate agent performance through multiple games."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def play_match(self, agent1_class: Type, agent2_class: Type, num_games: int = 100) -> Dict:
        """
        Play multiple games between two agent types.

        Args:
            agent1_class: Class of first agent
            agent2_class: Class of second agent
            num_games: Number of games to play

        Returns:
            Dictionary with match statistics
        """
        wins_agent1 = 0
        wins_agent2 = 0
        draws = 0
        total_rounds = 0

        start_time = time.time()

        for game_num in range(num_games):
            # Alternate starting positions for fairness
            if game_num % 2 == 0:
                agent1 = agent1_class()
                agent2 = agent2_class()
                agent1_is_player0 = True
            else:
                agent2 = agent2_class()
                agent1 = agent1_class()
                agent1_is_player0 = False

            game = Game(
                agent1 if agent1_is_player0 else agent2,
                agent2 if agent1_is_player0 else agent1,
                verbose=False
            )

            winner = game.play_game(max_rounds=10)
            total_rounds += game.state.round_count

            # Map winner back to agent1/agent2
            if winner == -1:
                draws += 1
            elif (winner == 0 and agent1_is_player0) or (winner == 1 and not agent1_is_player0):
                wins_agent1 += 1
            else:
                wins_agent2 += 1

            if self.verbose and (game_num + 1) % 10 == 0:
                print(f"  Completed {game_num + 1}/{num_games} games...")

        elapsed_time = time.time() - start_time

        return {
            'agent1_name': agent1_class.__name__,
            'agent2_name': agent2_class.__name__,
            'games_played': num_games,
            'agent1_wins': wins_agent1,
            'agent2_wins': wins_agent2,
            'draws': draws,
            'agent1_win_rate': wins_agent1 / num_games * 100,
            'agent2_win_rate': wins_agent2 / num_games * 100,
            'avg_rounds_per_game': total_rounds / num_games,
            'elapsed_time': elapsed_time,
        }

    def print_match_results(self, results: Dict):
        """Print formatted match results."""
        print(f"\n{'='*70}")
        print(f"Match: {results['agent1_name']} vs {results['agent2_name']}")
        print(f"{'='*70}")
        print(f"Games played: {results['games_played']}")
        print(f"{results['agent1_name']:20s}: {results['agent1_wins']:3d} wins ({results['agent1_win_rate']:5.1f}%)")
        print(f"{results['agent2_name']:20s}: {results['agent2_wins']:3d} wins ({results['agent2_win_rate']:5.1f}%)")
        print(f"{'Draws':20s}: {results['draws']:3d} ({results['draws']/results['games_played']*100:5.1f}%)")
        print(f"Average rounds per game: {results['avg_rounds_per_game']:.2f}")
        print(f"Time elapsed: {results['elapsed_time']:.2f}s")
        print(f"{'='*70}\n")

    def run_tournament(self, agent_classes: List[Type], games_per_match: int = 100):
        """
        Run a round-robin tournament with all agents.

        Args:
            agent_classes: List of agent classes to compete
            games_per_match: Number of games per matchup
        """
        print(f"\n{'#'*70}")
        print(f"# HANAMIKOJI TOURNAMENT")
        print(f"# {len(agent_classes)} agents, {games_per_match} games per matchup")
        print(f"{'#'*70}\n")

        # Track overall statistics
        stats = {agent_class.__name__: {'wins': 0, 'losses': 0, 'draws': 0}
                 for agent_class in agent_classes}

        all_results = []

        # Play all matchups
        for i, agent1_class in enumerate(agent_classes):
            for j, agent2_class in enumerate(agent_classes):
                if i >= j:  # Skip self-play and reverse matchups
                    continue

                print(f"Playing: {agent1_class.__name__} vs {agent2_class.__name__}")

                results = self.play_match(agent1_class, agent2_class, games_per_match)
                all_results.append(results)

                # Update overall stats
                stats[agent1_class.__name__]['wins'] += results['agent1_wins']
                stats[agent1_class.__name__]['losses'] += results['agent2_wins']
                stats[agent1_class.__name__]['draws'] += results['draws']

                stats[agent2_class.__name__]['wins'] += results['agent2_wins']
                stats[agent2_class.__name__]['losses'] += results['agent1_wins']
                stats[agent2_class.__name__]['draws'] += results['draws']

                self.print_match_results(results)

        # Print tournament summary
        self.print_tournament_summary(stats)
        self.print_head_to_head(agent_classes, all_results)

        return all_results, stats

    def print_tournament_summary(self, stats: Dict):
        """Print overall tournament standings."""
        print(f"\n{'='*70}")
        print(f"TOURNAMENT STANDINGS")
        print(f"{'='*70}")

        # Calculate win rates and sort
        standings = []
        for agent_name, record in stats.items():
            total_games = record['wins'] + record['losses'] + record['draws']
            win_rate = record['wins'] / total_games * 100 if total_games > 0 else 0
            standings.append({
                'name': agent_name,
                'wins': record['wins'],
                'losses': record['losses'],
                'draws': record['draws'],
                'total': total_games,
                'win_rate': win_rate,
            })

        # Sort by win rate
        standings.sort(key=lambda x: x['win_rate'], reverse=True)

        # Print table
        print(f"{'Rank':<6}{'Agent':<25}{'Wins':<8}{'Losses':<8}{'Draws':<8}{'Win Rate':<10}")
        print(f"{'-'*70}")

        for rank, agent_stats in enumerate(standings, 1):
            print(f"{rank:<6}{agent_stats['name']:<25}"
                  f"{agent_stats['wins']:<8}{agent_stats['losses']:<8}"
                  f"{agent_stats['draws']:<8}{agent_stats['win_rate']:>6.1f}%")

        print(f"{'='*70}\n")

    def print_head_to_head(self, agent_classes: List[Type], all_results: List[Dict]):
        """Print a head-to-head win-rate matrix (row vs column)."""
        names = [cls.__name__ for cls in agent_classes]

        # Build lookup: (row_name, col_name) -> row's win rate against col
        h2h: Dict[tuple, float] = {}
        for r in all_results:
            a1, a2 = r['agent1_name'], r['agent2_name']
            n = r['games_played']
            h2h[(a1, a2)] = r['agent1_wins'] / n * 100
            h2h[(a2, a1)] = r['agent2_wins'] / n * 100

        # Column width: enough for the longest name or "100.0%", whichever is wider
        cell_w = max(max(len(n) for n in names), 6) + 2  # +2 for padding
        row_label_w = max(len(n) for n in names) + 2

        # Header
        total_w = row_label_w + cell_w * len(names)
        print(f"\n{'='*total_w}")
        print("HEAD-TO-HEAD WIN RATES (row vs column)")
        print(f"{'='*total_w}")

        # Column headers
        header = " " * row_label_w + "".join(n.rjust(cell_w) for n in names)
        print(header)
        print("-" * total_w)

        # Rows
        for row in names:
            cells = []
            for col in names:
                if row == col:
                    cells.append("---".rjust(cell_w))
                else:
                    cells.append(f"{h2h[(row, col)]:.1f}%".rjust(cell_w))
            print(f"{row:<{row_label_w}}{''.join(cells)}")

        print(f"{'='*total_w}\n")


def main():
    """Run example tournament."""

    # List of agents to compete
    agents = [
        RandomAgent,
        GreedyAgent,
        BalancedAgent,
        CardCountingAgent,
    ]

    # Create evaluator
    evaluator = AgentEvaluator(verbose=True)

    # Run tournament
    results, stats = evaluator.run_tournament(agents, games_per_match=1000)

    # You can also run individual matches
    print("\nRunning focused match: CardCountingAgent vs GreedyAgent (5000 games)")
    result = evaluator.play_match(CardCountingAgent, GreedyAgent, num_games=5000)
    evaluator.print_match_results(result)


if __name__ == "__main__":
    main()
