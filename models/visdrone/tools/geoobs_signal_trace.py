"""Passive CPU diagnostics of actual observation and decoder returns."""
import torch

from projects.diffu_dino.modeling.flow_spatial_observation import FlowSpatialObservation
from projects.diffu_dino.modeling.dino_transformer import DINOTransformerDecoder
from tools.localization_trace import LocalizationTrace
from tools.selection_retention_trace import _require


def _cpu(value):
    return value.detach().cpu().clone()


class GeoObsSignalTrace(LocalizationTrace):
    def __init__(self, model):
        super().__init__(model)
        self._signal_decoder = model.transformer.decoder
        self._observation = getattr(self._signal_decoder, 'flow_spatial_observation', None)
        self.mode = getattr(model, 'flow_spatial_observation_mode',
                            getattr(self._observation, 'mode', 'off'))
        _require(self.mode in ('off', 'static', 'distribution'), 'invalid observation mode')
        _require(self.mode == 'off' or isinstance(self._observation, FlowSpatialObservation),
                 'enabled mode requires production observation')
        self.observations = []
        self.decoder = []
        self._observation_frame = self._signal_decoder_frame = None
        self._signal_decoder_count = 0
        self._observation_tags = None

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self._observation_frame = self._signal_decoder_frame = None
            self._observation_tags = None

    def _trace(self, frame, event, arg):
        code = frame.f_code
        if code is DINOTransformerDecoder.forward.__code__:
            if event == 'call' and frame.f_locals.get('self') is self._signal_decoder:
                _require(self._signal_decoder_frame is None, 'nested decoder calls')
                self._signal_decoder_count += 1
                self._signal_decoder_frame = frame
            elif event == 'return' and frame is self._signal_decoder_frame:
                if arg is not None:
                    _require(isinstance(arg, tuple) and len(arg) == 2, 'invalid decoder return')
                    self.decoder.append(dict(step_index=self._signal_decoder_count-1,
                                             states=_cpu(arg[0][-3:]), references=_cpu(arg[1][-3:])))
                self._signal_decoder_frame = None
            return self._trace
        if code is FlowSpatialObservation.forward.__code__:
            local = frame.f_locals
            if event == 'call' and local.get('self') is self._observation:
                _require(self._observation_frame is None, 'nested observation calls')
                self._observation_frame = frame
                caller = frame.f_back
                while caller is not None and caller is not self._signal_decoder_frame:
                    caller = caller.f_back
                self._observation_tags = dict(step_index=self._signal_decoder_count-1,
                                              layer_index=None if caller is None else caller.f_locals['layer_idx'])
            elif event == 'return' and frame is self._observation_frame:
                if arg is not None:
                    context = local['context']
                    record = dict(self._observation_tags)
                    for name in ('query', 'reference', 'state', 'points', 'valid', 'weights', 'pooled'):
                        record[name] = _cpu(local[name])
                    for name in ('time', 'history_valid', 'image_sizes', 'padded_image_size'):
                        record[name] = _cpu(context[name])
                    record['feature_hw'] = torch.tensor(context['feature'].shape[-2:])
                    history = context['history_quantiles']
                    record['history_quantiles'] = None if history is None else _cpu(history)
                    record['residual'] = _cpu(arg)
                    samples = _cpu(local['samples']).float()
                    mask = record['valid'][..., None]
                    count = mask.sum(-2).clamp_min(1)
                    mean = samples.masked_fill(~mask, 0).sum(-2) / count
                    record['sample_variance'] = ((samples-mean[..., None, :]).square().masked_fill(~mask,0).sum(-2)/count).mean(-1)
                    record['sample_energy'] = (samples.square().masked_fill(~mask,0).sum(-2)/count).mean(-1)
                    self.observations.append(record)
                self._observation_frame = None
                self._observation_tags = None
            return self._trace
        if event == 'return' and frame is self._endpoint_frame and arg is not None:
            state = frame.f_locals.get('layer_state')
            if state is not None:
                self._current['layer_state'] = _cpu(state)
        return super()._trace(frame, event, arg)

    def finish_signals(self):
        """Validate production cardinality and return independent CPU captures."""
        _require(self.verified and self._closed and not self._failed,
                 'finish_signals requires successful parent finish')
        _require(len(self.decoder) == self._signal_decoder_count == 3, 'expected three decoder returns')
        _require(len(self.observations) == (0 if self.mode == 'off' else 9), 'unexpected observation count')
        _require(len(self._localization) == 3, 'expected three endpoint returns')
        shapes = dict(query=(1,900,256), reference=(1,900,4), state=(1,900,4),time=(1,),
                      history_quantiles=(1,900,4,3),history_valid=(1,900),points=(1,900,36,2),
                      valid=(1,900,36),weights=(1,900,4,36),pooled=(1,900,64),residual=(1,900,256),
                      sample_variance=(1,900),sample_energy=(1,900), image_sizes=(1,2),
                      padded_image_size=(2,),feature_hw=(2,))
        def validate(record, expected):
            for name, shape in expected.items():
                value = record[name]
                if name == 'history_quantiles' and value is None:
                    continue
                _require(value.device.type == 'cpu' and tuple(value.shape) == shape
                         and torch.isfinite(value).all().item(), 'invalid signal shape/nonfinite '+name)
        for index, record in enumerate(self.observations):
            validate(record,shapes)
            _require(record['step_index'] == index//3 and record['layer_index'] == 3+index%3,
                     'unexpected observation step/layer order')
            _require(record['valid'].dtype == record['history_valid'].dtype == torch.bool, 'invalid validity dtype')
            invalid = ~record['valid']
            _require(not record['weights'].masked_select(invalid[:,:,None].expand_as(record['weights'])).any().item(),
                     'invalid points have nonzero attention')
            _require(not record['residual'][~record['valid'].any(-1)].any().item(), 'invalid query has nonzero residual')
        for index, record in enumerate(self.decoder):
            validate(record,dict(states=(3,1,900,256),references=(3,1,900,4)))
            _require(record['step_index'] == index, 'unexpected decoder step order')
        keys = ('class_logits','coarse_cxcywh','corner_logits','decoded_cxcywh','refined_cxcywh','layer_state')
        endpoints = [{key:_cpu(record[key]) for key in keys if key in record} for record in self._localization]
        for record in endpoints:
            validate(record, dict(coarse_cxcywh=(1,900,4),corner_logits=(1,900,132),
                                  decoded_cxcywh=(1,900,4),refined_cxcywh=(1,900,4)))
            _require(record['class_logits'].shape[:2] == (1,900) and torch.isfinite(record['class_logits']).all().item(), 'invalid endpoint classes')
            _require('layer_state' in record, 'missing endpoint layer_state')
            validate(record,dict(layer_state=(1,900,256)))
        def copy(records):
            return [{key:_cpu(value) if torch.is_tensor(value) else value for key,value in record.items()} for record in records]
        return dict(observations=copy(self.observations), decoder=copy(self.decoder), endpoints=endpoints)
