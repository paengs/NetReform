# NetReform
  - NetReform is a tensorflow wrapper for knowledge transfer. It is based on two papers; [Net2Net](http://arxiv.org/abs/1511.05641) and [Network Morphism](http://arxiv.org/abs/1603.01670).

## Dependencies
  - tensorflow
  - slim
  - numpy
  - scipy

## Usage
  - See example.py

    ```python
    from net_reform import NetReform
    # model, weights from a previous model
    nr = NetReform(model, weights, new_graph)
    # network reformation with random values
    out_graph = nr.reform()
    ```

## Notes
  - Currently, NetReform only supports Net2Net operators (wider & deeper). 
