""" 
Network Reformation Class

Author: Kyunghyun Paeng

"""
import numpy as np
import tensorflow as tf

from slim import ops
from slim import scopes
from slim import variables
from core.net2net import Net2Net
from core.net_morph import NetMorph

class NetReform(object):
    def __init__(self, teacher_model, teacher_weight, student_graph):
        self._n2n = Net2Net()
        self._nm = NetMorph()
        self._weight = teacher_weight
        self._conf = tf.ConfigProto(allow_soft_placement=True)
        self._sess = tf.Session(config=self._conf)
        self._load_teacher_net(teacher_model, teacher_weight)
        self._new_sess = tf.Session(config=self._conf, graph=student_graph)
        # temporary init & get vars to restore
        with self._new_sess.graph.as_default():
            self._new_sess.run(tf.initialize_all_variables())
            self._vars_to_restore = tf.get_collection(variables.VARIABLES_TO_RESTORE)
            self._vars_to_init = []
        # TODO: Assertion (vars_to_restore == new_vars)
        self._check_layers_to_restore()
        print '=== [Success] Network Reformation Initialization ==='

    @property
    def teacher_session(self):
        return self._sess
    @property
    def teacher_graph(self):
        return self._sess.graph
    @property
    def teacher_variables(self):
        return self._get_variables(self.teacher_graph)
    @property
    def student_session(self):
        return self._new_sess
    @property
    def student_graph(self):
        return self._new_sess.graph
    @property
    def student_variables(self):
        return self._get_variables(self.student_graph)

    def reform(self):
        """
        Network reformation with values computed from Net2Net or NetMorph

        """
        # compute new init values
        self.reform_rand()
        for v in self._vars_to_init:
            if 'weights' in v.name:
                if self._check_diff_from_name_or_not(v):
                    # Modify teacher's layers
                    if self._check_next_layer(v.name):
                        print v.name
                        self._update_layer(v.name, 'modify')
                else:
                    # Insert new layers
                    print v.name
                    self._update_layer(v.name, 'insert')

        return self.student_graph, self.student_session

    def reform_rand(self):
        """
        Network reformation with random values

        """
        saver = tf.train.Saver(self._vars_to_restore)
        saver.restore(self.student_session, self._weight)
        return self.student_graph, self.student_session

    def _check_next_layer(self, name):
        next_idx = self._get_layer_index(name, 'student') + 2
        for v in self._vars_to_init:
            if self._check_diff_from_name_or_not(v):
                if next_idx == self._get_layer_index(v.name, 'student'):
                    return True
        return False

    def _update_layer(self, name, mode):
        # TODO: if no bias settings, re-check layer index & 'weights'
        if mode == 'modify':
            target_idx = self._get_layer_index(name, 'teacher')
            update_idx = self._get_layer_index(name, 'student')
            new_width = self._get_value(update_idx, 'student').shape[-1]
            w1 = self._get_value(target_idx, 'teacher')
            b1 = self._get_value(target_idx+1, 'teacher')
            w2 = self._get_value(target_idx+2, 'teacher')
            nw1, nb1, nw2 = self._n2n.wider(w1, b1, w2, new_width, True)
            self.student_session.run(self.student_variables[update_idx].assign(nw1))
            self.student_session.run(self.student_variables[update_idx+1].assign(nb1))
            self.student_session.run(self.student_variables[update_idx+2].assign(nw2))
        elif mode == 'insert':
            target_idx = self._get_layer_index(name, 'student')
            w1 = self._get_value(target_idx-2, 'student')
            nw, nb = self._n2n.deeper(w1, True)
            self.student_session.run(self.student_variables[target_idx].assign(nw))
            self.student_session.run(self.student_variables[target_idx+1].assign(nb))
        
    def _grouping(self, teacher_vars, student_vars):
        assert len(teacher_vars) == len(student_vars), '[FAILED] Network Reform'
        group_list = []
        group = 0
        for i, idx in enumerate(student_vars):
            if i==0: 
                group_list.append(group)
            else:
                if idx-student_vars[i-1] != 1: 
                    group += 1
                group_list.append(group)
        assert len(group_list) == len(student_vars), '[FAILED] Grouping'
        return group_list
            
        

    def _check_layers_to_restore(self):
        from operator import itemgetter
        teacher_vars = self.teacher_variables
        student_vars = self.student_variables
        # Check layer name
        restore_idx = []
        for i, sv in enumerate(student_vars):
            for j, tv in enumerate(teacher_vars):
                if sv.name == tv.name:
                    restore_idx.append(i)
        if restore_idx:
            self._vars_to_restore = itemgetter(*restore_idx)(self._vars_to_restore)
        # Check layer shape
        restore_idx = []
        for i, nv in enumerate(self._vars_to_restore):
            tshape = self._get_value(nv.name, 'teacher').shape
            sshape = self._get_value(nv.name, 'student').shape
            if tshape == sshape:
                restore_idx.append(i)
        if restore_idx:
            self._vars_to_restore = itemgetter(*restore_idx)(self._vars_to_restore)
        # Set variables to initialize
        self._set_init_variables()   

    def _check_diff_from_name_or_not(self, var):
        for tv in self.teacher_variables:
            if var.name == tv.name:
                return True
        return False

    def _set_init_variables(self):
        for v in self.student_variables:
            check = False
            for r in self._vars_to_restore:
                if v.name == r.name:
                    check = True
                    break
            if not check:
                self._vars_to_init.append(v)

    def _get_layer_index(self, layer, mode):
        if mode == 'teacher':
            var_list = self.teacher_variables
        elif mode == 'student':
            var_list = self.student_variables
        else :
            assert False, 'Unknown mode'
        for i, v in enumerate(var_list):
            if v.name == layer:
                return i
        return None

    def _get_value(self, layer, mode):
        if mode == 'teacher':
            var_list = self.teacher_variables
            sess = self.teacher_session
        elif mode == 'student':
            var_list = self.student_variables
            sess = self.student_session
        else :
            assert False, 'Unknown mode'
        
        if type(layer) is int:
            return var_list[layer].eval(session=sess)
        else:
            for v in var_list:
                if v.name == layer:
                    return v.eval(session=sess)
        return None

    def _get_variables(self, graph):
        return graph.get_collection('trainable_variables')

    def _load_teacher_net(self, model, weight):
        saver = tf.train.import_meta_graph(model)
        saver.restore(self._sess, weight)
