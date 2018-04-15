#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <vector>
#include <cstdlib>
#include <assert.h>
#include <algorithm> //sort
#include <limits>
#include <numeric> //accumulate

using namespace std;

class TwoMeansTreeNode {
	private:
		TwoMeansTreeNode *left;
		TwoMeansTreeNode *right;
		unsigned int depth;
		vector< vector<double> > means;
		//double means[2];
		vector< vector<double> > points;
		int ndimensions;

	public: 
		TwoMeansTreeNode(vector< vector<double> > pts, unsigned int d, bool isLeafNode){
			if(isLeafNode){
			    //set points at this node
			    for(int i=0; i<pts.size();i++){
			    	points.push_back(pts[i]);
			    } 
			    //cout << "added points to leaf node"<<endl;
			} else {
				assert(pts.size() == 2);
				assert((pts[0]).size() == (pts[1]).size());
				means.push_back(pts[0]);
				means.push_back(pts[1]);
			}
			left = NULL;
			right = NULL;
			depth = d;
		} 
		
		~TwoMeansTreeNode(){
			delete this->right;
			delete this->left;
		}
		
		void setRightChild(TwoMeansTreeNode* t1){
			this->right = t1;
		}
	
		void setLeftChild(TwoMeansTreeNode* t2){
			this->left = t2;
		}
	
		unsigned int getDepth(){
			return this->depth;
		}
		
		TwoMeansTreeNode* getLeftChild(){
			return this->left;
		}
		
		TwoMeansTreeNode* getRightChild(){
			return this->right;
		}
	
		vector< vector<double> > getPoints(){
			return this->points;
		}
		
		vector< vector<double> > getMeans(){
			return this->means;
		}
		
};

void printTree(TwoMeansTreeNode *tree){
	assert( (tree->getLeftChild()== NULL && tree->getRightChild() == NULL ) || (tree->getLeftChild()!= NULL && tree->getRightChild() != NULL));
	if( tree->getLeftChild() == NULL && tree->getRightChild()==NULL ){
		vector< vector<double> > pts = tree->getPoints();
		cout << "Leaf node points at depth "<< tree->getDepth() <<": <";
		for(int i=0; i<pts.size(); i++){
			cout << "(";
			for(int j=0; j<(pts[i]).size(); j++){
				cout << ", " << (pts[i])[j];
			}
			cout <<"), ";
		}
		cout <<">"<<endl;
	} else {
		vector< vector<double> > means = tree->getMeans();
		//cout <<"Internal node at depth " << tree->getDepth() << " means: "<<means[0]<<", "<<means[1]<<endl;
		unsigned int depthcurrent = tree->getDepth();
		//if(depthcurrent==0){
			cout << depthcurrent<<"--";
		//}
		printTree(tree->getLeftChild());
		cout <<"--and---";
		printTree(tree->getRightChild());		
	}
	return;
}

double euclideanDistance(vector< double > a, vector< double > b){
	assert(a.size() == b.size());
	vector<double> sqdists(a.size(), 0.0);
	for(int i=0; i<a.size(); i++){
		sqdists[i] = (a[i]-b[i])*(a[i]-b[i]);
	}
	double dist = sqrt(accumulate(sqdists.begin(), sqdists.end(), 0.0));
	return dist;
}

void printLeafNodes(TwoMeansTreeNode *tree){	
	if( tree->getLeftChild() == NULL && tree->getRightChild()==NULL ){
		vector< vector<double> > pts = tree->getPoints();
		cout << "Leaf node points:\n "<<endl;
		for(int i=0; i<pts.size(); i++){
			cout << "(";
			for(int j=0; j<(pts[i]).size(); j++){
				if(j<(pts[i]).size()-1){
					cout << pts[i][j] <<",";
				} else {
					cout << pts[i][j] <<")";
				}
			}
			if(i<pts.size()-1){
				cout << ", ";		
			}
		}
		cout <<endl;
	} else {
		printLeafNodes(tree->getLeftChild());
		printLeafNodes(tree->getRightChild());		
	}
	return;
}

vector<double> twoMeansOneD(vector<double> X){
	int npts = X.size();
	sort(X.begin(), X.end());
	bool isCloserToMean1[npts];
	vector<double> means;
	vector<double>::iterator it=X.begin();
	//initialize the two means
	double mean1=0.0, mean2=0.0, sum1=0.0, sum2=0.0;
	//double mean1=X.begin(), mean2=0.0, sum1=X.begin(), sum2=0.0;	
	vector<double>::iterator it2 = X.begin();
	while(it2<X.end()){
		sum2 += *it2;
		it2++;
	}
	
	int seenpts=0, nSwaps=0;
	double sumsqdists=0.0, minsumsqdists=numeric_limits<double>::max();
	while(seenpts<(npts-1)){
		nSwaps=0;
		it++;
		seenpts++;
		// re-compute means
		sum1+=*it;
		sum2-=*it;
		mean1 = sum1/(double) seenpts;
		mean2 = sum2/(double) (npts-seenpts);
		//cout << "iteration "<<seenpts<<": means found: mean1="<<mean1<<", mean2="<<mean2<<endl;
		//cout << " sum1="<<sum1<<", sum2="<<sum2<<endl;	
		// assign points to closest mean
		for(int i=0; i<npts; i++){
			double dist1 = fabs(X[i] - mean1);
			double dist2 = fabs(X[i] - mean2);
			if(dist1<dist2){
				if(isCloserToMean1[i] == false) nSwaps++;
				isCloserToMean1[i] = true;
			} else {
				if(isCloserToMean1[i] == true) nSwaps++;
				isCloserToMean1[i] = false;	
			}
		}
		//cout<<"iteration "<< seenpts <<": nSwaps="<<nSwaps<<endl;
	
		// store means if sum of squared distances
		// to means yields best split
		sumsqdists=0.0;
		for(int i=0; i<npts; i++){
			if(isCloserToMean1[i]){
			   sumsqdists += fabs(X[i]-mean1)*fabs(X[i]-mean1);
			} else {
			   sumsqdists += fabs(X[i]-mean2)*fabs(X[i]-mean2);
			}
		}
		
		if(sumsqdists<minsumsqdists){
			minsumsqdists = sumsqdists;
			means.clear();
			means.push_back(mean1);
			means.push_back(mean2);
		}
	}	
	//cout << "returning means" <<means[0]<<" and "<<means[1]<<endl;
	return means;
}

vector< vector<double> > twomeans(vector< vector<double> > X){	
	int npts = X.size();
	int ndims=0;
	if(!X.empty()){
		ndims = X[0].size();
	} 
	bool isCloserToMean1[npts];	
	//choose starting means at random from the points in the data set
	int idx1 = rand() % npts;
	int idx2 = rand() % npts;	
	vector<double> mean1 = X[idx1];
	vector<double> mean2 = X[idx2];
	//keep track of how many swaps happened 
	// between this iteration and the last
	int nSwaps = npts;

	while (nSwaps>0){ //iterate until no points are re-assigned
		//iteratively update means and re-assign points
		nSwaps = 0;
		// assign points to closest mean
		for(int i=0; i<npts; i++){
			double dist1 = euclideanDistance(X[i], mean1);
			double dist2 = euclideanDistance(X[i], mean2);
			if(dist1<dist2){
				if(isCloserToMean1[i] == false) nSwaps++;
				isCloserToMean1[i] = true;
			} else {
				if(isCloserToMean1[i] == true) nSwaps++;
				isCloserToMean1[i] = false;	
			}
		}
		// re-compute means
		mean1.clear();
		mean1.resize(ndims, 0.0);
		mean2.clear();
		mean2.resize(ndims, 0.0);
		double npts1 = 0.0, npts2 = 0.0;
		for(int i=0; i<npts; i++){
			if(isCloserToMean1[i]){ 
				for(int dim=0; dim<ndims; dim++){
					mean1[dim] += (X[i])[dim];
				}
				npts1++;
			} else { 
				for(int dim=0; dim<ndims; dim++){
					mean2[dim] += (X[i])[dim];
				}
				npts2++;
			}
		}
		for(int dim=0; dim<ndims; dim++){
			mean1[dim] = mean1[dim]/npts;
		}
		for(int dim=0; dim<ndims; dim++){
			mean2[dim] = mean2[dim]/npts;
		}
	}
	vector< vector<double> > means;
	means.push_back(mean1);
	means.push_back(mean2);
	//cout << "found means "<<mean1<<" and "<<mean2<<endl;
	return means;
}
	
void splitDataBy2Means(vector<vector<double> > X, vector< vector<double> > mus, bool *closerToMu1){
	int n = X.size();
	int ndims=0;
	if(!X.empty()){
		ndims = X[0].size();
	}
	double d_mu1, d_mu2;
	for(int i=0; i<n; i++){
		d_mu1 = euclideanDistance(X[i] , mus[0]); 
		d_mu2 = euclideanDistance(X[i] , mus[1]);
		if(d_mu1 < d_mu2){
			closerToMu1[i] = true;	
		} else { // in this case d_mu1 >= d_mu2
			closerToMu1[i] = false;
		} 
	}
}

TwoMeansTreeNode* buildTwoMeansTree(vector< vector<double> > X, unsigned int d, unsigned int depth_threshold){
	int npts = X.size();
	// split criteria
	//if(npts <= 4){ //stop splitting when number of points in a node is low
	if(d>=depth_threshold || npts <=2){
		TwoMeansTreeNode* leafnode = new TwoMeansTreeNode(X, d, true);
		return leafnode;
	}
	bool closertoMu1[npts];
	vector< vector<double> > means = twomeans(X);//twoMeansOneD(X);
 	splitDataBy2Means(X, means, closertoMu1);
	//cout << "split data of size "<<npts<<"by 2-means."<<endl;	
	vector< vector<double> > leftsplit;
	vector< vector<double> > rightsplit;
	int nleft=0, nright=0;
	for(int i=0; i<X.size(); i++){
		if(closertoMu1[i]){
			leftsplit.push_back(X[i]);
			nleft++;
		} else {
			rightsplit.push_back(X[i]);
			nright++;
		}
	}
	//cout << "splitting: "<<nleft<<" left and "<<nright<<" right"<<endl;
	TwoMeansTreeNode* leftsubtree = buildTwoMeansTree(leftsplit, d+1, depth_threshold);
	TwoMeansTreeNode* rightsubtree = buildTwoMeansTree(rightsplit, d+1, depth_threshold);
	TwoMeansTreeNode* root = new TwoMeansTreeNode(means, d, false);
	root->setLeftChild(leftsubtree);
	//cout << "set left child "<<endl;
	root->setRightChild(rightsubtree);
	//cout << "set right child"<<endl;
	return root;	
}

/* numPoints: returns the number of points
	stored in the leaves of a TwoMeansTreeNode 
	tree
*/
int numPoints(TwoMeansTreeNode* tree){
	if(tree->getLeftChild() == NULL && tree->getRightChild()==NULL){
		vector< vector<double> > leafpoints = tree->getPoints();
		return leafpoints.size();
	} else {
		return (numPoints(tree->getLeftChild()) + numPoints(tree->getRightChild()));
	}
}

vector< TwoMeansTreeNode* > buildRandomForest(vector< vector<double> > X, int numTrees, unsigned int depthThreshold){
	
	vector< TwoMeansTreeNode* > forest;
	for(int i=0; i<numTrees; i++){
		TwoMeansTreeNode* tree = buildTwoMeansTree(X, 0, depthThreshold);
		forest.push_back(tree);
		cout << "finished tree "<<i<<endl;
	}
	return forest;
}
	
bool appearInSameLeafNode(vector<double> a, vector<double> b, TwoMeansTreeNode* tree){
	if(tree->getLeftChild() == NULL && tree->getRightChild()==NULL){
		vector< vector<double> > pointsInLeafNode = tree->getPoints();
		return ( ( find(pointsInLeafNode.begin(), pointsInLeafNode.end(),a)!=pointsInLeafNode.end() ) && ( find(pointsInLeafNode.begin(), pointsInLeafNode.end(), b)!=pointsInLeafNode.end() ) );
	} else {
		return (appearInSameLeafNode( a,b,tree->getLeftChild() ) || appearInSameLeafNode( a,b,tree->getRightChild() ) );
	}
}

// test driver function
int main(int argc, char **argv){
	unsigned int treedepth;
	
	if(argc < 4)
	{
		printf("Usage : ./testTwoMeansForest <number of dimensions> <tree depth> <data set size (number of points)>\n");
		exit(-1);
	}
	
	treedepth = (unsigned int)atoi(argv[2]);
	int datasetsize = atoi(argv[3]);

	// small simple data test
	/*
	int Xarr[] = {1, 5, 7.8, 10.2, 15, 19, 21, 199, 200, 201, 202, 203, 204, 205}; 
	vector<double> X(Xarr, Xarr + sizeof(Xarr)/sizeof(Xarr[0]));
	TwoMeansTreeNode* tree = buildTwoMeansTree(X, 0, 4);
	cout << "Tree 1: "<< endl;
	printTree(tree);
	*/
		
	vector< vector<double> > Y;
	int ndims = atoi(argv[1]);
	for(int i=0; i<datasetsize; i++){
		vector<double> temp;
		for(int j=0; j<ndims; j++){
			temp.push_back((double) rand() / RAND_MAX);
		}
		Y.push_back(temp);
		temp.clear();
	}
	TwoMeansTreeNode* tree2 = buildTwoMeansTree(Y, 0, treedepth);
	cout << "Tree 2: (random numbers between 0 and 1)"<< endl;
	printTree(tree2);	
	cout << "Tree 2: number of points in tree = "<<numPoints(tree2)<<endl;
	int ntrees = 100;
	vector< TwoMeansTreeNode* > random2meansforest = buildRandomForest(Y,ntrees, treedepth);
	
	// print out pairwise estimated similarities as well as true distances
	double estimated_sim_ij=0.0;
	stringstream ofss;
	ofss<<"truedist_vs_estimatedsim_"<<datasetsize<<"_pts"
		<<ndims<<"dimensions_depth"<<treedepth<<".txt";
	ofstream true_est_comparefile;
	true_est_comparefile.open(ofss.str().c_str());
	double true_dist_ij;
	for(int i=0; i<datasetsize; i++){
		for(int j=0; j<datasetsize; j++){
			estimated_sim_ij=0;
			for(int k=0; k<ntrees; k++){	
				if(appearInSameLeafNode(Y[i],Y[j],random2meansforest[k])){
					estimated_sim_ij++;
					/*cout << "found "<<Y[i]<<" and "<<Y[j]
						<< " in same node in tree"
						<<k<<endl;*/
				}
			}
			estimated_sim_ij /= (double) ntrees;
			true_dist_ij = euclideanDistance(Y[i],Y[j]);
			true_est_comparefile << true_dist_ij
				<<"\t"<<estimated_sim_ij
				<<endl;
		}
		cout <<"finished printing similarities and true distances for point "<<i<<": (";
		for(int d=0; d<ndims; d++){
			cout<<Y[i][d]<<",";
		}
		cout <<")";
	}
	true_est_comparefile.close();

	/*	
	vector<double> Z;
	for(int i=0; i<5000; i++){
		Z.push_back(i);	
	}
	TwoMeansTreeNode* tree3 = buildTwoMeansTree(Z, 0, 8);
	cout << "Tree 3: "<<endl;
	printTree(tree3);	
	*/

	/* normally distributed data */
	/*vector<double> N;
	for(int i=0; i<100; i++){
		N.push_back(i);	
	}
	TwoMeansTreeNode* treeN = buildTwoMeansTree(N, 0, 8);
	printTree(treeN);	
	*/
	
	return 0;
}
