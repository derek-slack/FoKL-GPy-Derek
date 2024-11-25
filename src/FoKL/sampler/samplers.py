__all__ = ["BaseSampler", "GibbsSampler", "GibbsUpdateSampler"]

class BaseSampler:
    def __init__(self, inputs, data, **kwargs):
        self.inputs = inputs
        self.data = data
        self.kwargs = kwargs

    def sample(self):
        raise NotImplementedError("Sample method must be implemented by subclasses.")

    @staticmethod
    def create_sampler(samplers_dict, sampler_name, *args, **kwargs):
        # Dynamically create a sampler instance based on the sampler name
        if sampler_name in samplers_dict:
            sampler_class = samplers_dict[sampler_name]
            return sampler_class(*args, **kwargs)
        else:
            raise ValueError(f"Sampler '{sampler_name}' not found.")


class GibbsSampler(BaseSampler):
    def sample(self):
        from gibbs import gibbs  # Import dynamically
        return gibbs(self.inputs, self.data, **self.kwargs)


class GibbsUpdateSampler(BaseSampler):
    def sample(self):
        from gibbs_Xin_update import gibbs_Xin_update  # Import dynamically
        return gibbs_Xin_update(self.inputs, self.data, **self.kwargs)


# Dictionary to map sampler names to classes
SAMPLERS_DICT = {
    "gibbs": GibbsSampler,
    "gibbs_update": GibbsUpdateSampler
}