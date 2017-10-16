import logging
import os

import numpy as np
from numpy import corrcoef

from scipy.cluster.hierarchy import linkage
from scipy.misc import comb

from pandas import DataFrame

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib
from sklearn.tree.tests.test_tree import y_random
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plottingscripts.plotting.scatter import plot_scatter_plot

from aslib_scenario.aslib_scenario import ASlibScenario
from autofolio.selector.pairwise_classification import PairwiseClassifier
from autofolio.selector.classifiers.random_forest import RandomForest

from spysmac.asapy.utils.util_funcs import get_cdf_x_y

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "MIT"
__email__ = "lindauer@cs.uni-freiburg.de"


class FeatureAnalysis(object):

    def __init__(self,
                 output_dn: str,
                 scenario,
                 feat_names):
        '''
        Constructor
        Arguments
        ---------
        output_dn:str
            output directory name
        '''
        self.logger = logging.getLogger("Feature Analysis")
        self.scenario = scenario
        self.feature_data = {}
        for name in feat_names:
            insts = self.scenario.train_insts
            insts.extend(self.scenario.train_insts)
            self.feature_data[name] = {}
            for i in insts:
                self.feature_data[name][i] = self.scenario.feature_dict[i][feat_names.index(name)]

        self.output_dn = os.path.join(output_dn, "feature_plots")
        if not os.path.isdir(self.output_dn):
            os.makedirs(self.output_dn)

    def get_box_violin_plots(self,
            feat_names):
        '''
            for each feature generate a plot with box and vilion plot

            Parameters
            ----------
            feat_names: list[str]
                names of the features

            Returns
            -------
            list of tuples of feature name and feature plot file name
        '''
        self.logger.info("Plotting box and violin plots........")

        files_ = []

        for feat_name in sorted(feat_names):
            matplotlib.pyplot.close()
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 5))
            vec = self.scenario.feature_data[feat_name].values
            vec = vec[~np.isnan(vec)]
            axes[0].violinplot(
                [vec], showmeans=False, showmedians=True, vert=False)
            axes[0].xaxis.grid(True)
            plt.setp(axes[0], yticks=[1], yticklabels=[""])
            axes[1].boxplot(vec, vert=False)
            axes[1].xaxis.grid(True)
            plt.setp(axes[1], yticks=[1], yticklabels=[""])

            plt.tight_layout()

            out_fn = os.path.join(
                self.output_dn, "violin_box_%s_plot.png" % (feat_name.replace("/", "_")))
            plt.savefig(out_fn)
            files_.append((feat_name, out_fn))

        return files_

    def correlation_plot(self):
        '''
            generate correlation plot using spearman correlation coefficient and ward clustering
            Returns
            -------
            file name of saved plot
        '''
        matplotlib.pyplot.close()
        self.logger.info("Plotting correlation plots........")

        feature_data = self.scenario.feature_data
        features = list(self.scenario.feature_data.columns)
        feature_data = feature_data.fillna(feature_data.mean())
        feature_data = feature_data.values

        n_features = len(features)

        data = np.zeros((n_features, n_features)) + 1  # similarity
        for i in range(n_features):
            for j in range(i + 1, n_features):
                rho = corrcoef([feature_data[:, i], feature_data[:, j]])[0, 1]
                if np.isnan(rho):  # is nan if one feature vec is constant
                    rho = 0
                data[i, j] = rho
                data[j, i] = rho

        link = linkage(data * -1, 'ward')  # input is distance -> * -1

        sorted_features = [[a] for a in features]
        for l in link:
            new_cluster = sorted_features[int(l[0])][:]
            new_cluster.extend(sorted_features[int(l[1])][:])
            sorted_features.append(new_cluster)

        sorted_features = sorted_features[-1]

        # resort data
        indx_list = []
        for f in features:
            indx_list.append(sorted_features.index(f))
        indx_list = np.argsort(indx_list)
        data = data[indx_list, :]
        data = data[:, indx_list]

        fig, ax = plt.subplots()
        heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

        plt.xlim(0, data.shape[0])
        plt.ylim(0, data.shape[0])

        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(sorted_features, minor=False)
        ax.set_yticklabels(sorted_features, minor=False)
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=45, fontsize=2, ha="left")
        labels = ax.get_yticklabels()
        plt.setp(labels, rotation=0, fontsize=2, ha="right")

        fig.colorbar(heatmap)

        plt.tight_layout()

        out_plot = os.path.join(
            self.output_dn, "correlation_plot_features.png")
        plt.savefig(out_plot, format="png", dpi=400)

        return out_plot

    def feature_importance(self):
        '''
            train pairwise random forests and average the feature importance from all trees
        '''
        matplotlib.pyplot.close()
        self.logger.info("Plotting feature importance........")
        self.scenario.feature_data = self.scenario.feature_data.fillna(
            self.scenario.feature_data.mean())

        pc = PairwiseClassifier(classifier_class=RandomForest)
        config = {}
        config["rf:n_estimators"] = 100
        config["rf:max_features"] = "auto"
        config["rf:criterion"] = "gini"
        config["rf:max_depth"] = None
        config["rf:min_samples_split"] = 2
        config["rf:min_samples_leaf"] = 1
        config["rf:bootstrap"] = True
        pc.fit(scenario=self.scenario, config=config)

        importances = [
            rf.model.feature_importances_ for rf in pc.classifiers if np.isnan(rf.model.feature_importances_).sum() == 0]
        median_importance = np.median(importances, axis=0)
        q25 = np.percentile(importances, 0.25, axis=0)
        q75 = np.percentile(importances, 0.75, axis=0)

        feature_names = np.array(self.scenario.feature_data.columns)

        # sort features by average importance and look only at the first 15
        # features
        N_FEAT = min(feature_names.shape[0], 15)
        indices = np.argsort(median_importance)[::-1]
        median_importance = median_importance[indices][:N_FEAT]
        q25 = q25[indices][:N_FEAT]
        q75 = q75[indices][:N_FEAT]
        feature_names = feature_names[indices[:N_FEAT]]

        plt.figure()
        # only the first 10 most important features
        plt.bar(range(N_FEAT), median_importance,
                color="r", yerr=[q25, q75], align="center")

        plt.xlim([-1, N_FEAT])
        plt.xticks(range(N_FEAT), feature_names, rotation=40, ha='right')
        plt.tight_layout()
        out_fn = os.path.join(self.output_dn, "feature_importance.png")
        plt.savefig(out_fn, format="png")

        return out_fn

    def cluster_instances(self):
        '''
            use pca to reduce feature dimensions to 2 and cluster instances using k-means afterwards
        '''
        matplotlib.pyplot.close()
        self.logger.info("Plotting clusters........")
        # impute missing data; probably already done, but to be on the safe
        # side
        self.scenario.feature_data = self.scenario.feature_data.fillna(
            self.scenario.feature_data.mean())

        # feature data
        features = self.scenario.feature_data.values

        # scale features
        ss = StandardScaler()
        features = ss.fit_transform(features)

        # feature reduction: pca
        pca = PCA(n_components=2)
        features = pca.fit_transform(features)

        # cluster with k-means
        scores = []
        for n_clusters in range(2, 12):
            km = KMeans(n_clusters=n_clusters)
            y_pred = km.fit_predict(features)
            score = silhouette_score(features, y_pred)
            scores.append(score)

        best_score = min(scores)
        best_run = scores.index(best_score)
        n_clusters = best_run + 2
        km = KMeans(n_clusters=n_clusters)
        y_pred = km.fit_predict(features)

        plt.figure()
        plt.scatter(features[:, 0], features[:, 1], c=y_pred)

        plt.tight_layout()
        out_fn = os.path.join(self.output_dn, "feature_clusters.png")
        plt.savefig(out_fn, format="png")

        return out_fn

    def get_bar_status_plot(self):
        '''
            get status distribution as stacked bar plot
        '''
        matplotlib.pyplot.close()
        self.logger.info("Plotting bar plots........")
        runstatus_data = self.scenario.feature_runstatus_data

        width = 0.5
        stati = ["ok", "timeout", "memout",
                 "presolved", "crash", "other", "unknown"]

        count_stats = np.array(
            [runstatus_data[runstatus_data == status].count().values for status in stati])
        count_stats = count_stats / len(self.scenario.instances)

        colormap = plt.cm.gist_ncar
        cc = [colormap(i) for i in np.linspace(0, 0.9, len(stati))]

        bottom = np.zeros((len(runstatus_data.columns)))
        ind = np.arange(len(runstatus_data.columns)) + 0.5
        plots = []
        for id, status in enumerate(stati):
            plots.append(
                plt.bar(ind, count_stats[id, :], width, color=cc[id], bottom=bottom))
            bottom += count_stats[id, :]

        plt.ylabel('Frequency of runstatus')
        plt.xticks(
            ind + width / 2., list(runstatus_data.columns), rotation=45, ha="right")
        lgd = plt.legend(list(map(lambda x: x[0], plots)), stati, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                         ncol=3, mode="expand", borderaxespad=0.)

        plt.tight_layout()
        out_fn = os.path.join(self.output_dn, "status_bar_plot.png")
        plt.savefig(out_fn, bbox_extra_artists=(lgd,), bbox_inches='tight')

        return out_fn

    def get_feature_cost_cdf_plot(self):
        '''
            get cdf for feature costs
        '''

        matplotlib.pyplot.close()
        self.logger.info("Plotting feature cost cdfs plots........")

        if self.scenario.feature_cost_data is None:
            raise("Feature cost not provided")

        from cycler import cycler

        gs = matplotlib.gridspec.GridSpec(1, 1)

        fig = plt.figure()
        ax1 = plt.subplot(gs[0:1, :])

        colormap = plt.cm.gist_ncar
        fig.gca().set_prop_cycle(cycler('color', [
            colormap(i) for i in np.linspace(0, 0.9, len(self.scenario.algorithms))]))

        if self.scenario.features_cutoff_time:
            max_val = self.scenario.features_cutoff_time
        else:
            max_val = self.scenario.feature_cost_data.max().max()

        self.scenario.feature_cost_data[
            self.scenario.feature_cost_data == 0] = max_val

        min_val = max(0.0005, self.scenario.feature_cost_data.min().min())

        for step in self.scenario.feature_steps:
            x, y = get_cdf_x_y(self.scenario.feature_cost_data[step], max_val)
            ax1.step(x, y, label=step)

        ax1.grid(
            True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax1.set_xlabel("Cost")
        ax1.set_ylabel("P(x<X)")
        ax1.set_xlim([min_val, max_val])
        ax1.set_xscale('log')

        #ax1.legend(loc='lower right')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        out_fn = os.path.join(self.output_dn, "cdf_plot.png")

        plt.savefig(out_fn, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, pad_inches=0.02, bbox_inches='tight')
        return out_fn
