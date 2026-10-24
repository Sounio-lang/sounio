// Independent HC3 numerical reference, C++23 + libquadmath.
// Build:
//   g++ -std=c++23 -O2 -Wall -Wextra -pedantic linear_hc3_reference.cpp -lquadmath -o linear_hc3_reference
// Execution receipts are recorded externally; source comments are not receipts.
// Production API: stats::regression::robust_covariance::multiple_linear_hc3,
// stride8, n1..256,p1..8,p<=n; method3, assumptions_assessed=false.
// Independence: raw-design two-pass modified Gram-Schmidt in __float128,
// triangular inverse, and explicit observationwise influence outer products.
// No Householder implementation, production kernel, or floating normal-equation
// inversion is reused. Rank comparisons are column-relative MGS diagnostics;
// cases are clearly separated, not claims of identical CPQR threshold decisions.
// Final binary64 representability is checked. Production can additionally refuse
// an unrepresentable internal normalized coordinate/solve; this independent raw
// quad algorithm does not imitate those implementation-specific intermediates.
//
// IMPORTANT residual semantics:
//  raw_quad_residual = input b - fitted value from the computed quad QR.
//  reported_residual = binary64(input b - binary64(quad fitted value)).
// HC3 uses the latter, matching the public b-minus-reported-fit measurand.
// Production and oracle reported fits may differ in their last bits. Therefore
// tiny residuals/covariances need backward-error-aware absolute comparisons.
// Separate analytic-perfect records do not replace either computed residual.
// This tool emits numeric reference data; it is not an automated Sounio test.
#include <quadmath.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <utility>
using Q = __float128;
using Vec = std::vector<Q>;
using Matrix = std::vector<Vec>;
struct Case {
    std::string name;
    int n=0,p=0;
    double rcond=0;
    std::array<double,2048> a{};
    std::array<double,256> b{};
    std::vector<Q> analytic_beta; // Additional exact binary-rational certificate.
};
struct Result {
    int fit_status=0,covariance_status=0,rank=0;
    Vec beta,fitted,raw_residual,reported_residual,oracle_leverage;
    Matrix covariance;
    Vec standard_errors,leverage;
    std::string note="ordinary";
};
Q dot(const Vec&a,const Vec&b) {
    Q sum=0;for(std::size_t i=0;i<a.size();++i)sum+=a[i]*b[i];
    return sum;
}
bool finite_double(Q x) { return std::isfinite(static_cast<double>(x)); }
bool strict_double(Q x) {
    return finite_double(x)&&(x==0||static_cast<double>(x)!=0);
}
void clear_covariance(Result&r,int status,int n,int p) {
    r.covariance_status=status;
    r.covariance.assign(p,Vec(p));
    r.standard_errors.assign(p,0);
    r.leverage.assign(n,0);
}
Result evaluate(const Case&c) {
    Result r;
    auto fit_error=[&](int status) {
        r.fit_status=status;
        r.covariance_status=1;
        r.note="point_fit_unavailable";
        return r;
    };
    if(c.n<1||c.n>256||c.p<1||c.p>8||c.p>c.n)return fit_error(1);
    if(!std::isfinite(c.rcond)||c.rcond<0||c.rcond>=1)return fit_error(5);
    for(int i=0;i<c.n;++i) {
        if(!std::isfinite(c.b[i]))return fit_error(2);
        for(int j=0;j<c.p;++j)
            if(!std::isfinite(c.a[i*8+j]))return fit_error(2);
    }
    Matrix q(c.p,Vec(c.n)),upper(c.p,Vec(c.p));
    const Q cutoff=c.rcond==0
        ?static_cast<Q>(c.n)*static_cast<Q>(std::numeric_limits<double>::epsilon())
        :static_cast<Q>(c.rcond);
    for(int j=0;j<c.p;++j) {
        Vec column(c.n);
        for(int i=0;i<c.n;++i)column[i]=static_cast<Q>(c.a[i*8+j]);
        const Q original=sqrtq(dot(column,column));
        for(int pass=0;pass<2;++pass)for(int k=0;k<j;++k) {
            const Q projection=dot(q[k],column);
            upper[k][j]+=projection;
            for(int i=0;i<c.n;++i)column[i]-=projection*q[k][i];
        }
        const Q norm=sqrtq(dot(column,column));
        if(original==0||norm<=cutoff*original)return fit_error(3);
        if(!finiteq(norm)||norm<=0)return fit_error(4);
        upper[j][j]=norm;
        for(int i=0;i<c.n;++i)q[j][i]=column[i]/norm;
        ++r.rank;
    }
    Matrix inverse(c.p,Vec(c.p));
    for(int col=0;col<c.p;++col)for(int row=c.p-1;row>=0;--row) {
        Q value=row==col?1:0;
        for(int k=row+1;k<c.p;++k)value-=upper[row][k]*inverse[k][col];
        inverse[row][col]=value/upper[row][row];
        if(!finiteq(inverse[row][col]))return fit_error(4);
    }
    Vec rhs(c.p),y(c.n);
    for(int i=0;i<c.n;++i)y[i]=static_cast<Q>(c.b[i]);
    for(int j=0;j<c.p;++j)rhs[j]=dot(q[j],y);
    r.beta.assign(c.p,0);
    for(int j=0;j<c.p;++j)r.beta[j]=dot(inverse[j],rhs);
    for(Q value:r.beta)if(!strict_double(value))return fit_error(4);
    r.fitted.assign(c.n,0);r.raw_residual.assign(c.n,0);
    r.reported_residual.assign(c.n,0);r.oracle_leverage.assign(c.n,0);
    for(int i=0;i<c.n;++i) {
        for(int j=0;j<c.p;++j) {
            r.fitted[i]+=static_cast<Q>(c.a[i*8+j])*r.beta[j];
            r.oracle_leverage[i]+=q[j][i]*q[j][i];
        }
        if(!finite_double(r.fitted[i]))return fit_error(4);
        r.raw_residual[i]=y[i]-r.fitted[i];
        const double reported_fit=static_cast<double>(r.fitted[i]);
        const double reported_residual=c.b[i]-reported_fit;
        if(!std::isfinite(reported_residual))return fit_error(4);
        r.reported_residual[i]=static_cast<Q>(reported_residual);
    }
    clear_covariance(r,0,c.n,c.p);
    if(c.n<=c.p) {
        clear_covariance(r,2,c.n,c.p);r.note="no_residual_degrees_of_freedom";
        return r;
    }
    for(int i=0;i<c.n;++i) {
        Q h=r.oracle_leverage[i];
        if(!finiteq(h)||h<0) {
            clear_covariance(r,4,c.n,c.p);return r;
        }
        // No epsilon clipping: h>=1 is unavailable even with zero residual.
        if(h>=1) {
            clear_covariance(r,3,c.n,c.p);r.note="unit_leverage";
            return r;
        }
    }
    for(int i=0;i<c.n;++i) {
        const Q h=r.oracle_leverage[i];
        const Q adjusted=r.reported_residual[i]/(1-h);
        Vec influence(c.p);
        for(int j=0;j<c.p;++j) {
            Q g=0;
            for(int k=0;k<c.p;++k)g+=inverse[j][k]*q[k][i];
            influence[j]=g*adjusted;
            if(!strict_double(influence[j])) {
                clear_covariance(r,4,c.n,c.p);return r;
            }
        }
        for(int j=0;j<c.p;++j)for(int k=0;k<c.p;++k)
            r.covariance[j][k]+=influence[j]*influence[k];
    }
    for(int j=0;j<c.p;++j)for(int k=0;k<c.p;++k)
        if(!finite_double(r.covariance[j][k])) {
            clear_covariance(r,4,c.n,c.p);return r;
        }
    for(int j=0;j<c.p;++j) {
        r.standard_errors[j]=sqrtq(r.covariance[j][j]);
        // Diagonal underflow is unavailable; signed off-diagonal IEEE zero is
        // permitted, matching the public availability distinction.
        if(!strict_double(r.covariance[j][j])||
           !strict_double(r.standard_errors[j])) {
            clear_covariance(r,4,c.n,c.p);return r;
        }
    }
    r.leverage=r.oracle_leverage;
    if(!c.analytic_beta.empty())r.note="analytic_perfect_record_separate_from_numeric_QR";
    return r;
}
Case dense(std::string name,std::vector<std::vector<double>>rows,
           std::vector<double>response) {
    Case c;c.name=std::move(name);c.n=static_cast<int>(rows.size());
    c.p=rows.empty()?0:static_cast<int>(rows[0].size());
    for(int i=0;i<c.n;++i) {
        c.b[i]=response[i];
        for(int j=0;j<c.p;++j)c.a[i*8+j]=rows[i][j];
    }
    return c;
}
std::vector<Case> cases() {
    Case ordinary=dense("heteroscedastic",{{1,0},{1,1},{1,2},{1,3},
        {1,4},{1,5},{1,6},{1,7}},
        {1.2,2.7,5.4,6.6,9.6,10.4,13.5,14.5});
    std::vector<Case> result{ordinary};
    result.push_back(dense("high_leverage",{{1,0},{1,1},{1,2},
        {1,3},{1,4},{1,1000}},{1,3,4,8,7,2004}));
    result.push_back(dense("heteroscedastic_large_tail",{{1,0},{1,1},
        {1,2},{1,3},{1,4},{1,5},{1,6},{1,7}},
        {1,3,5,7,9,20,-10,50}));
    for(double scale:{1e-150,1e150}) {
        Case c=ordinary;c.name=scale<1?"response_tiny":"response_huge";
        for(int i=0;i<c.n;++i)c.b[i]*=scale;
        result.push_back(c);
        c=ordinary;c.name=scale<1?"design_tiny":"design_huge";
        for(int i=0;i<c.n;++i)for(int j=0;j<c.p;++j)c.a[i*8+j]*=scale;
        result.push_back(c);
    }
    Case mixed=ordinary;mixed.name="mixed_column_units";
    for(int i=0;i<mixed.n;++i) {
        mixed.a[i*8]*=1e-150;mixed.a[i*8+1]*=1e150;
    }
    result.push_back(mixed);
    Case permuted=ordinary;permuted.name="column_permutation";
    for(int i=0;i<permuted.n;++i)
        std::swap(permuted.a[i*8],permuted.a[i*8+1]);
    result.push_back(permuted);
    Case rows=ordinary;rows.name="row_permutation";
    for(int i=0;i<rows.n/2;++i) {
        std::swap(rows.b[i],rows.b[rows.n-1-i]);
        for(int j=0;j<rows.p;++j)
            std::swap(rows.a[i*8+j],rows.a[(rows.n-1-i)*8+j]);
    }
    result.push_back(rows);
    Case perfect=dense("analytic_perfect",{{1,0},{1,1},{1,2},
        {1,3},{1,4}},{2,5,8,11,14});
    perfect.analytic_beta={2,3};result.push_back(perfect);
    result.push_back(dense("zero_response",{{1,0},{1,1},{1,2},
        {1,3},{1,4}},{0,0,0,0,0}));
    result.push_back(dense("duplicate_columns",{{1,1},{2,2},
        {3,3},{4,4}},{1,3,4,8}));
    result.push_back(dense("zero_column",{{1,0},{1,0},
        {1,0},{1,0}},{1,3,4,8}));
    result.push_back(dense("unit_leverage_singleton",{{1,0},{1,0},
        {0,1}},{1,3,8}));
    result.push_back(dense("square_no_df",{{1,0},{0,1}},{2,3}));
    result.push_back(dense("one_row_no_df",{{1}},{2}));
    result.push_back(dense("no_intercept",{{1},{2},{3},{4}},
        {1,5,4,9}));
    Case largest;largest.name="max_256_by_8";largest.n=256;largest.p=8;
    for(int i=0;i<largest.n;++i) {
        double response=static_cast<double>((i*7)%13-6)*0.125;
        for(int j=0;j<largest.p;++j) {
            unsigned bits=static_cast<unsigned>(i&j);
            unsigned parity=0;while(bits){parity^=bits&1U;bits>>=1U;}
            largest.a[i*8+j]=parity?-1:1;
            response+=largest.a[i*8+j]*static_cast<double>(j+1);
        }
        largest.b[i]=response;
    }
    result.push_back(largest);
    Case poison=ordinary;poison.name="inactive_nonfinite";
    poison.a[2047]=std::numeric_limits<double>::quiet_NaN();
    poison.b[255]=std::numeric_limits<double>::infinity();
    result.push_back(poison);
    for(double scale:{1e-200,1e200}) {
        Case c=ordinary;c.name=scale<1?"covariance_underflow":"covariance_overflow";
        for(int i=0;i<c.n;++i)c.b[i]*=scale;
        result.push_back(c);
    }
    Case invalid=ordinary;invalid.name="active_nonfinite";
    invalid.a[0]=std::numeric_limits<double>::quiet_NaN();
    result.push_back(invalid);
    invalid=ordinary;invalid.name="invalid_rcond";invalid.rcond=-1;
    result.push_back(invalid);
    return result;
}
void number(Q value) {
    // Preserve quad values and binary64 rounding separately; no hidden
    // substitution of overflow/underflow with a plausible finite constant.
    char buffer[128];quadmath_snprintf(buffer,sizeof buffer,"%.36Qe",value);
    std::cout<<buffer<<"\t";
    const double rounded=static_cast<double>(value);
    if(!std::isfinite(rounded))std::cout<<"unrepresentable";
    else std::cout<<std::scientific<<std::setprecision(17)<<rounded;
}
void entry(const std::string&kind,const Case&c,int i,int j,Q value) {
    std::cout<<kind<<"\t"<<c.name<<"\t"<<i<<"\t"<<j<<"\t";
    number(value);std::cout<<"\n";
}
int main(int argc,char**) {
    if(argc!=1) {
        std::cerr<<"Usage: linear_hc3_reference > hc3-reference.tsv\n";
        return 2;
    }
    std::cout<<"# schema=linear-hc3-reference-v1\n"
        <<"# numeric columns: kind,case,index_i,index_j,quad_value,binary64_value\n"
        <<"# case columns: case,name,n,p,fit_status,covariance_status,rank,note\n"
        <<"# covariance uses reported_residual; raw_quad_residual stays separate\n"
        <<"# no t/p/CI, clustering, or empirical assumption assessment\n";
    for(const auto&c:cases()) {
        const Result r=evaluate(c);
        std::cout<<"case\t"<<c.name<<"\t"<<c.n<<"\t"<<c.p<<"\t"
            <<r.fit_status<<"\t"<<r.covariance_status<<"\t"<<r.rank
            <<"\t"<<r.note<<"\n";
        if(r.fit_status==0) {
            for(int j=0;j<c.p;++j)entry("coefficient",c,j,-1,r.beta[j]);
            for(int i=0;i<c.n;++i) {
                entry("fitted_quad",c,i,-1,r.fitted[i]);
                entry("raw_quad_residual",c,i,-1,r.raw_residual[i]);
                entry("reported_residual",c,i,-1,r.reported_residual[i]);
                entry("oracle_leverage_unrounded",c,i,-1,r.oracle_leverage[i]);
                entry("leverage_payload",c,i,-1,r.leverage[i]);
            }
            for(int j=0;j<c.p;++j) {
                entry("standard_error",c,j,-1,r.standard_errors[j]);
                for(int k=0;k<c.p;++k)
                    entry("covariance",c,j,k,r.covariance[j][k]);
            }
        }
        if(!c.analytic_beta.empty()) {
            // Exact finite binary-rational fixture certificate is additional
            // output only. It never alters MGS beta, residuals, or covariance.
            bool exact=true;
            for(int i=0;i<c.n;++i) {
                Q fit=0;
                for(int j=0;j<c.p;++j)
                    fit+=static_cast<Q>(c.a[i*8+j])*c.analytic_beta[j];
                if(fit!=static_cast<Q>(c.b[i]))exact=false;
            }
            if(!exact) {
                std::cerr<<"Invalid analytic certificate: "<<c.name<<"\n";
                return 3;
            }
            for(int j=0;j<c.p;++j) {
                entry("analytic_perfect_coefficient",c,j,-1,c.analytic_beta[j]);
                for(int k=0;k<c.p;++k)
                    entry("analytic_perfect_covariance",c,j,k,0);
            }
        }
    }
    return 0;
}
