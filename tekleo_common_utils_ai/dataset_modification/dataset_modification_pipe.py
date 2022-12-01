from typing import List
import random
from tekleo_common_message_protocol import OdSample
from tekleo_common_utils_ai.dataset_modification.abstract_dataset_modifier import AbstractDatasetModifier


class BehaviorRandom:
    ALL = 'all' # apply all modifiers
    RANDOM = 'random' # apply random modifiers


class BehaviorChaining:
    CHAINED = 'chained' # chained to produce a single image
    UNIQUE = 'unique' # each modifier produces a new image


class BehaviorOriginals:
    KEEP = 'keep' # keep the originals
    OVERWRITE = 'overwrite' # overwrite the originals


class DatasetModificationPipe:
    # behavior_dataset_ratio_to_process -> 1.0 work on all images | 0.15 work on 15% of images
    # behavior_random -> "all" apply all modifiers | "random" apply random modifiers
    # behavior_chaining -> "chained" chained to produce a single image | "unique" each modifier produces a new image
    # behavior_originals -> "keep" keep the originals | "overwrite" overwrite the originals
    def __init__(
            self, modifiers: List[AbstractDatasetModifier],
            behavior_dataset_ratio_to_process: float, behavior_random_seed: int,
            behavior_random: BehaviorRandom, behavior_chaining: BehaviorChaining, behavior_originals: BehaviorOriginals
    ):
        self.modifiers = modifiers
        self.behavior_dataset_ratio_to_process = behavior_dataset_ratio_to_process
        self.behavior_random_seed = behavior_random_seed
        self.behavior_random = behavior_random # apply all modifiers | apply random modifiers
        self.behavior_chaining = behavior_chaining # chained to produce a single image | each modifier produces a new image
        self.behavior_originals = behavior_originals # keep the originals | overwrite the originals

        # Instance of random generator
        self.random = random.Random()
        self.random.seed(self.behavior_random_seed)

    def process(self, samples: List[OdSample]) -> List[OdSample]:
        new_samples = []

        # Determine which samples to process
        samples_to_process = samples.copy()
        if self.behavior_dataset_ratio_to_process != 1.0:
            num_of_samples_to_process = int(self.behavior_dataset_ratio_to_process * len(samples))
            samples_to_process = self.random.shuffle(samples_to_process)
            samples_to_process = samples_to_process[0:num_of_samples_to_process]

        for sample in samples_to_process:
            # Determine which modifiers to apply to the sample
            modifiers_to_apply = []
            for modifier in self.modifiers:
                if self.behavior_random == BehaviorRandom.RANDOM:
                    should_apply = self.random.choice([True, False])
                    if should_apply:
                        modifiers_to_apply.append(modifier)
                elif self.behavior_random == BehaviorRandom.ALL:
                    modifiers_to_apply.append(modifier)

            # Keep modifiers chained
            if self.behavior_chaining == BehaviorChaining.CHAINED:
                new_sample = sample
                for modifier in modifiers_to_apply:
                    new_sample = modifier.apply(new_sample)
                new_samples.append(new_sample)

            # Produce unique image for each modifier
            elif self.behavior_chaining == BehaviorChaining.UNIQUE:
                for modifier in modifiers_to_apply:
                    new_sample = modifier.apply(sample)
                    new_samples.append(new_sample)

        # Add originals to the dataset
        if self.behavior_originals == BehaviorOriginals.KEEP:
            new_samples.extend(samples)

        return new_samples

