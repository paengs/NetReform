# NetReform
NetReform is a tensorflow wrapper for knowledge transfer. It is based on two papers; [Net2Net](http://arxiv.org/abs/1511.05641) and [Network Morphism](http://arxiv.org/abs/1603.01670).

## Dependencies
  - tensorflow
  - [slim](https://github.com/paengs/NetReform/tree/master/slim) (modified ver.)
  - numpy
  - scipy

## Usage
  - See [example.py](https://github.com/paengs/NetReform/blob/master/example.py)

    ```python
    from net_reform import NetReform
    # model, weights from a previous model
    nr = NetReform(model, weights, new_graph)
    out_graph, out_session = nr.reform_rand() # NetReform with random values
    out_graph, out_session = nr.reform() # NetReform with values derived from net2net or netmorph func.
    ...
    obj = out_graph.get_collection('objective')[0]
    out_session.run(obj)
    ```

## Notes
  - Currently, NetReform only supports Net2Net operators (wider & deeper). 
