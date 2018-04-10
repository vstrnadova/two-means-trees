#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <cstdlib>
#include <assert.h>
#include <algorithm> //sort
#include <limits>

using namespace std;

class TwoMeansTreeNode {
	private:
		TwoMeansTreeNode *left;
		TwoMeansTreeNode *right;
		int depth;
		double means[2];
		vector< double > points;

	public: 
		TwoMeansTreeNode(vector<double> pts, int d, bool isLeafNode){
			if(isLeafNode){
			    //set points at this node
			    for(int i=0; i<pts.size();i++){
			    	points.push_back(pts[i]);
			    } 
			    //cout << "added points to leaf node"<<endl;
			} else {
				assert(pts.size() == 2);
				means[0] = pts[0];
				means[1] = pts[1];
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
	
		int getDepth(){
			return this->depth;
		}
		
		TwoMeansTreeNode* getLeftChild(){
			return this->left;
		}
		
		TwoMeansTreeNode* getRightChild(){
			return this->right;
		}
	
		vector<double> getPoints(){
			return this->points;
		}
		
		vector<double> getMeans(){
			vector<double> mus;
			mus.push_back(this->means[0]);
			mus.push_back(this->means[1]);
			return mus;
		}
		
};

void printTree(TwoMeansTreeNode *tree){
	assert( (tree->getLeftChild()== NULL && tree->getRightChild() == NULL ) || (tree->getLeftChild()!= NULL && tree->getRightChild() != NULL));
	if( tree->getLeftChild() == NULL && tree->getRightChild()==NULL ){
		vector<double> pts = tree->getPoints();
		cout << "Leaf node at depth "<< tree->getDepth() <<" points:\n "<<endl;
		for(int i=0; i<pts.size(); i++){
			cout << ", " << pts[i];
		}
		cout <<endl;
	} else {
		vector<double> means = tree->getMeans();
		//cout <<"Internal node at depth " << tree->getDepth() << " means: "<<means[0]<<", "<<means[1]<<endl;
		printTree(tree->getLeftChild());
		printTree(tree->getRightChild());		
	}
	return;
}

void printLeafNodes(TwoMeansTreeNode *tree){	
	if( tree->getLeftChild() == NULL && tree->getRightChild()==NULL ){
		vector<double> pts = tree->getPoints();
		cout << "Leaf node points:\n "<<endl;
		for(int i=0; i<pts.size(); i++){
			cout << ", " << pts[i] ;
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

vector<double> twomeans(vector<double> X){	
	int npts = X.size();
	bool isCloserToMean1[npts];	
	//choose starting means at random from the points in the data set
	int idx1 = rand() % npts;
	int idx2 = rand() % npts;	
	double mean1 = X[idx1];
	double mean2 = X[idx2];
	//keep track of how many swaps happened 
	// between this iteration and the last
	int nSwaps = npts;

	while (nSwaps>0){ //iterate until no points are re-assigned
		//iteratively update means and re-assign points
		nSwaps = 0;
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
		// re-compute means
		mean1 = 0.0;
		mean2 = 0.0;
		double npts1 = 0.0, npts2 = 0.0;
		for(int i=0; i<npts; i++){
			if(isCloserToMean1[i]){ 
				mean1+=X[i];
				npts1++;
			} else { 
				mean2 += X[i];
				npts2++;
			}
		}
		mean1 = mean1/npts1;
		mean2 = mean2/npts2;
	}
	vector<double> means;
	means.push_back(mean1);
	means.push_back(mean2);
	//cout << "found means "<<mean1<<" and "<<mean2<<endl;
	return means;
}
	
void splitDataBy2Means(vector<double> X, vector<double> mus, bool *closerToMu1){
	int n = X.size();
	double d_mu1, d_mu2;
	for(int i=0; i<n; i++){
		d_mu1 = fabs(X[i] - mus[0]); 
		d_mu2 = fabs(X[i] - mus[1]);
		if(d_mu1 < d_mu2){
			closerToMu1[i] = true;	
		} else { // in this case d_mu1 >= d_mu2
			closerToMu1[i] = false;
		} 
	}
}

TwoMeansTreeNode* buildTwoMeansTree(vector<double> X, int d, int depth_threshold){
	int npts = X.size();
	// split criteria
	//if(npts <= 4){ //stop splitting when number of points in a node is low
	if(d>=depth_threshold || npts <=2){
		TwoMeansTreeNode* leafnode = new TwoMeansTreeNode(X, d, true);
		return leafnode;
	}
	bool closertoMu1[npts];
	vector<double> means = twomeans(X);//twoMeansOneD(X);
 	splitDataBy2Means(X, means, closertoMu1);
	//cout << "split data of size "<<npts<<"by 2-means."<<endl;	
	vector<double> leftsplit;
	vector<double> rightsplit;
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
		vector<double> leafpoints = tree->getPoints();
		return leafpoints.size();
	} else {
		return (numPoints(tree->getLeftChild()) + numPoints(tree->getRightChild()));
	}
}

vector< TwoMeansTreeNode* > buildRandomForest(vector<double> X, int numTrees){
	
	int depthThreshold=8;
	vector< TwoMeansTreeNode* > forest;
	for(int i=0; i<numTrees; i++){
		TwoMeansTreeNode* tree = buildTwoMeansTree(X, 0, depthThreshold);
		forest.push_back(tree);
		cout << "finished tree "<<i<<endl;
	}
	return forest;
}
	
bool appearInSameLeafNode(double a, double b, TwoMeansTreeNode* tree){
	if(tree->getLeftChild() == NULL && tree->getRightChild()==NULL){
		vector<double> pointsInLeafNode = tree->getPoints();
		return ( find(pointsInLeafNode.begin(), pointsInLeafNode.end(),a)!=pointsInLeafNode.end() && find(pointsInLeafNode.begin(), pointsInLeafNode.end(), b)!=pointsInLeafNode.end() );
	} else {
		return (appearInSameLeafNode(a,b,tree->getLeftChild()) || appearInSameLeafNode(a,b,tree->getRightChild()) );
	}
}

// test driver function
int main(){
	cout << "testing..."<<endl;
	// small simple data test
	/*
	int Xarr[] = {1, 5, 7.8, 10.2, 15, 19, 21, 199, 200, 201, 202, 203, 204, 205}; 
	vector<double> X(Xarr, Xarr + sizeof(Xarr)/sizeof(Xarr[0]));
	TwoMeansTreeNode* tree = buildTwoMeansTree(X, 0, 4);
	cout << "Tree 1: "<< endl;
	printTree(tree);
	*/
		
	vector<double> Y;
	int datasetsize=500;
	for(int i=0; i<datasetsize; i++){
		Y.push_back((double) rand() / RAND_MAX);
	}
	TwoMeansTreeNode* tree2 = buildTwoMeansTree(Y, 0, 8);
	cout << "Tree 2: (random numbers between 0 and 1)"<< endl;
	printTree(tree2);	
	cout << "Tree 2: number of points in tree = "<<numPoints(tree2)<<endl;
	int ntrees = 100;
	vector< TwoMeansTreeNode* > random2meansforest = buildRandomForest(Y,ntrees);
	
	// print out pairwise estimated similarities as well as true distances
	double estimated_sim_ij=0.0;
	ofstream true_est_comparefile;
	true_est_comparefile.open("truedist_vs_estimatedsim.txt");
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
			true_dist_ij = fabs(Y[i]-Y[j]);
			true_est_comparefile << true_dist_ij
				<<"\t"<<estimated_sim_ij
				<<endl;
		}
		cout <<"finished printing similarities and true distances for point "<<i<<": "<<Y[i]<<endl;
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
