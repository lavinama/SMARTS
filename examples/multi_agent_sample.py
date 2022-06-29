import pathlib

import gym

from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.sstudio import build_scenario
from smarts.zoo.agent_spec import AgentSpec


class SimpleAgent(Agent):
    def act(self, obs):
        return "keep_lane"

def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
        agent_builder=SimpleAgent,
    )

    agent_specs = {
        "Agent-007": agent_spec,
        "Agent-008": agent_spec,
    }
    
    # make env
    env = gym.make(
        "smarts.env:hiway-v0", # env entry name
        scenarios=scenarios, # a list of paths to folders of scenarios
        agent_specs=agent_specs, #  dictionary of agents to interact with the environment
        # headless=False, # headless mode. False to enable Envision visualization of the environment
        visdom=True, # Visdom visualization of observations. False to disable. only supported in HiwayEnv.
        seed=42, # RNG Seed, seeds are set at the start of simulation, and never automatically re-seeded.
    )

    # reset env and build agents
    observations = env.reset()
    agents = {
        agent_id: agent_spec.build_agent()
        for agent_id, agent_spec in agent_specs.items()
    }
    
    # step env
    for _ in range(1000):
        # Instead of writing:
        ### agent_obs = observations[AGENT_ID]
        ### agent_action = agent.act(agent_obs)
        # Write:
        agent_actions = {
            agent_id: agents[agent_id].act(agent_obs) for agent_id, agent_obs in observations.items()
        }
        observations, rewards, dones, infos_for_active_agents = env.step(agent_actions)
    
    # close env
    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("multi-agent-example")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(pathlib.Path(__file__).absolute().parents[1] / "scenarios" / "loop")
        ]

    build_scenario(args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
    )
