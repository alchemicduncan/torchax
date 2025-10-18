# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools

import torch
from time import time
from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration
from torch.utils import _pytree as pytree
from PIL import Image
import requests

import torchax
from torchax import tensor
from torchax.interop import torch_view

import jax
import torch.func


class CompiledModule:

  def __init__(self, model):
    weights = model.state_dict()
    weights.update(model.named_parameters())
    self._weights = pytree.tree_map_only(torch.Tensor,
                                         torchax.tensor.move_to_device, weights)
    self._model = model

    self._func_jitted_torch = None

  def _maybe_move_tensor(self, tensor):
    if isinstance(
        tensor, torch.Tensor) and not isinstance(tensor, torchax.tensor.Tensor):
      return torchax.tensor.move_to_device(tensor)
    return tensor

  def _make_jitted(self, args, kwargs):
    static = []
    for i, a in enumerate(args):
      if not isinstance(a, torch.Tensor):
        static.append(i + 1)  # weight is 0
    static_argnames = []
    for k, v in kwargs.items():
      if not isinstance(v, torch.Tensor):
        static_argnames.append(k)

    def f(weights, *args, **kwargs):
      weights, args, kwargs = tensor.wrap((weights, args, kwargs))
      with tensor.XLAFunctionMode(), tensor.XLADispatchMode(
      ):
        res = torch.func.functional_call(self._model, weights, args, kwargs)
        if isinstance(res, tuple) and len(res) == 1:
          res = res[0]
      return tensor.unwrap(res)

    fjit = jax.jit(f, static_argnames=tuple(static_argnames))
    return torch_view(fjit)

  def forward(self, *args, **kwargs):
    (args, kwargs) = pytree.tree_map(self._maybe_move_tensor, (args, kwargs))
    if self._func_jitted_torch is None:
      self._func_jitted_torch = self._make_jitted(args, kwargs)
    return self._func_jitted_torch(self._weights, *args, **kwargs)

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def __getattr__(self, key):
    return getattr(self._model, key)


def compile_model(model):
  return CompiledModule(model)


def main():
  model_id = "GSAI-ML/LLaDA-V"
  config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
  model = LlavaForConditionalGeneration.from_pretrained(
      model_id,
      config=config,
      dtype=torch.float16,
      trust_remote_code=True,
  )
  processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

  model = compile_model(model)

  # Prepare image and text
  url = "https://www.ilankelman.org/stopsigns/australia.jpg"
  image = Image.open(requests.get(url, stream=True).raw)
  prompt = "<image>\nWhat is unusual about this image?"

  inputs = processor(text=prompt, images=image, return_tensors="pt")


  global_bs = 1
  inference_steps = 50 # max_new_tokens
  print(
      f'global batch size {global_bs}',
      f'inference steps {inference_steps}',
      flush=True)

  iters = 5
  for i in range(iters):
    start = time()
    # Generate
    generate_ids = model.generate(**inputs, max_new_tokens=inference_steps)
    generated_text = processor.batch_decode(generate_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print(f'Step {i} inference time {time()-start} sec', flush=True)
    print(f"Generated text: {generated_text}")


if __name__ == '__main__':
  main()
